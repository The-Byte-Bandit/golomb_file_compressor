import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from flask_bcrypt import Bcrypt
import jwt
from functools import wraps
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import json



# ================================================================
#  CONFIGURATION
# ================================================================
load_dotenv()
app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure upload folder exists
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize DB and bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Create database if not exists
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
if not database_exists(engine.url):
    create_database(engine.url)


# ================================================================
#  DATABASE MODELS
# ================================================================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CompressedFile(db.Model):
    __tablename__ = 'compressed_files'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)
    m = db.Column(db.Integer, default=10)
    rows = db.Column(db.Integer)
    cols = db.Column(db.Integer)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    file_metadata = db.Column(db.Text, nullable=True)
    original_size_bits = db.Column(db.Text, nullable=True)
    compressed_size_bits = db.Column(db.Text, nullable=True)
    compression_ratio = db.Column(db.Text, nullable=True)


# ================================================================
#  JWT AUTH MIDDLEWARE
# ================================================================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'message': 'Invalid token user'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401

        return f(current_user, *args, **kwargs)
    return decorated


# ================================================================
#  GOLOMB ENCODING / DECODING LOGIC
# ================================================================



# ---------------------- Golomb Encoding / Decoding ----------------------

def golomb_encode_number(n, m):
    """
    Encode a single non-negative integer n using Golomb coding with parameter m.

    Parameters:
        n (int): The integer to encode (must be non-negative)
        m (int): The Golomb parameter (positive integer)

    Returns:
        str: Binary string representing the Golomb code
    """
    if n < 0:
        raise ValueError("Golomb coding only supports non-negative integers")
    if m <= 0:
        raise ValueError("Parameter m must be a positive integer")

    q = n // m
    r = n % m

    unary = '1' * q + '0'
    b = int(np.ceil(np.log2(m)))
    cutoff = 2 ** b - m

    if r < cutoff:
        binary = format(r, f'0{b-1}b')
    else:
        r += cutoff
        binary = format(r, f'0{b}b')

    return unary + binary


def golomb_encode(array, m):
    """
    Encode a NumPy array using Golomb coding.

    Parameters:
        array (np.ndarray): Array of non-negative integers
        m (int): Golomb parameter

    Returns:
        bytes: Byte representation of the encoded bitstring
    """
    flat = array.flatten(order='C')
    bitstring = ''.join([golomb_encode_number(int(x), m) for x in flat])

    # Pad to full bytes
    if len(bitstring) % 8 != 0:
        bitstring += '0' * (8 - (len(bitstring) % 8))

    encoded_bytes = int(bitstring, 2).to_bytes(len(bitstring) // 8, byteorder='big')
    return encoded_bytes


def golomb_decode_bitstream(bitstream, m):
    """
    Decode a Golomb-encoded bitstream (string of bits) into integers.

    Parameters:
        bitstream (str): Binary string representing encoded data
        m (int): Golomb parameter

    Returns:
        list[int]: List of decoded integers
    """
    i = 0
    decoded = []
    b = int(np.ceil(np.log2(m)))
    cutoff = 2 ** b - m

    while i < len(bitstream):
        # Read unary part (quotient)
        q = 0
        while i < len(bitstream) and bitstream[i] == '1':
            q += 1
            i += 1
        if i >= len(bitstream):
            break
        i += 1  # skip the '0'

        # Read remainder part
        if i + b - 1 > len(bitstream):
            break
        r_bits = bitstream[i:i + b - 1]
        r = int(r_bits, 2)
        i += b - 1

        # Handle cutoff region
        if r >= cutoff:
            if i >= len(bitstream):
                break
            if i + 1 > len(bitstream):
                break
            next_bit = bitstream[i]
            r = (r << 1) | int(next_bit, 2)
            r -= cutoff
            i += 1

        decoded.append(q * m + r)

    return decoded


def golomb_decode_bytes(encoded_bytes, m, rows, cols):
    """
    Decode bytes back into a NumPy matrix using Golomb decoding.

    Parameters:
        encoded_bytes (bytes): Encoded byte stream
        m (int): Golomb parameter
        rows (int): Number of rows in the output matrix
        cols (int): Number of columns in the output matrix

    Returns:
        np.ndarray: Decoded NumPy matrix
    """
    bitstring = bin(int.from_bytes(encoded_bytes, "big"))[2:].zfill(len(encoded_bytes) * 8)
    decoded = golomb_decode_bitstream(bitstring, m)

    if len(decoded) < rows * cols:
        raise ValueError("Decoded data is smaller than expected matrix size")

    matrix = np.array(decoded[:rows * cols]).reshape((rows, cols), order='C')
    return matrix


# ================================================================
#  AUTH ROUTES
# ================================================================

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"error": "Username, email, and password are required"}), 400

    # Check duplicates
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({"error": "Username or email already exists"}), 400

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_pw)

    db.session.add(new_user)
    db.session.commit()

    # Generate JWT token
    token = jwt.encode(
        {"user_id": new_user.id, "exp": datetime.utcnow() + timedelta(hours=24)},
        app.config['SECRET_KEY'],
        algorithm="HS256"
    )

    return jsonify({
        "username": new_user.username,
        "email": new_user.email,
        "token": token
    }), 201


# ----------------------------
# LOGIN ROUTE
# ----------------------------
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    identifier = data.get('username') or data.get('email')
    password = data.get('password')

    if not identifier or not password:
        return jsonify({"error": "Username/email and password required"}), 400

    # Allow login with either email or username
    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode(
        {"user_id": user.id, "exp": datetime.utcnow() + timedelta(hours=24)},
        app.config['SECRET_KEY'],
        algorithm="HS256"
    )

    return jsonify({
        "username": user.username,
        "email": user.email,
        "token": token
    }), 200


# ================================================================
#  FILE ROUTES
# ================================================================
# ================================================================
#  FILE ROUTES (MERGED FUNCTIONALITY)
# ================================================================
@app.route("/compress", methods=["POST"])
@token_required
def compress_file(current_user):
    """Compress CSV or NPY file using Golomb coding."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename
    # Read file directly into memory
    file_content = file.read()
    # file.save(filepath)

    try:
        ext = os.path.splitext(filename)[1].lower()

        # ---------- Handle CSV ----------
        # if ext == ".csv":
            # with open(filepath, "r") as f:
            #     sample = f.readline().strip()

            # # detect delimiter
            # if "\t" in sample:
            #     delimiter = "\t"
            # elif "," in sample:
            #     delimiter = ","
            # else:
            #     delimiter = None

            # # detect header
            # has_header = not all(ch.isdigit() or ch.isspace() or ch in [delimiter, ".", "-"] for ch in sample)
            # df = pd.read_csv(filepath, delimiter=delimiter, header=0 if has_header else None)
            # data = df.to_numpy(dtype=int)
            # columns = df.columns.tolist() if has_header else None

        if ext == ".csv":
            # Use io.StringIO to read CSV from bytes
            file_stream = io.StringIO(file_content.decode('utf-8'))
            
            # Detect delimiter from first line
            sample = file_stream.readline()
            file_stream.seek(0)
            
            if "\t" in sample:
                delimiter = "\t"
            elif "," in sample:
                delimiter = ","
            else:
                delimiter = None

            has_header = not all(ch.isdigit() or ch.isspace() or ch in [delimiter, ".", "-"] for ch in sample if ch.strip())
            
            df = pd.read_csv(file_stream, delimiter=delimiter, header=0 if has_header else None)
            data = df.to_numpy(dtype=int)
            columns = df.columns.tolist() if has_header else None
        # ---------- Handle NPY ----------
        # elif ext == ".npy":
        #     data = np.load(filepath)
        #     if data.ndim == 1:
        #         data = data.reshape(-1, 1)
        #     delimiter = "\t"
        #     columns = None
        #     has_header = False

        elif ext == ".npy":
            # Load NPY from bytes
            file_bytes = io.BytesIO(file_content)
            data = np.load(file_bytes)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            delimiter = "\t"
            columns = None
            has_header = False
        else:
            return jsonify({"error": f"Unsupported file type '{ext}'"}), 400

        rows, cols = data.shape
        m = int(request.form.get("m", 10))

        # --- Encode data ---
        encoded_bytes = golomb_encode(data, m)

        # --- Save compressed data to DB ---
        file_metadata = {
            "delimiter": delimiter or ",",
            "has_header": has_header,
            "columns": columns,  # list or None
            "original_ext": ext,
        }

       
        # new_file = CompressedFile(
        #     filename=filename,
        #     data=encoded_bytes,
        #     m=m,
        #     rows=rows,
        #     cols=cols,
        #     uploaded_by=current_user.id
        # )
        # db.session.add(new_file)
        # db.session.commit()

        # --- Compute compression ratio ---
        original_bits = data.size * 32  # assuming int32 original
        compressed_bits = len(encoded_bytes) * 8
        compression_ratio = round(original_bits / compressed_bits, 3)


         # Save to DB
        new_file = CompressedFile(
            filename=filename,
            data=encoded_bytes,
            m=m,
            rows=rows,
            cols=cols,
            uploaded_by=current_user.id,
            file_metadata=json.dumps(file_metadata),
            original_size_bits = original_bits,
            compressed_size_bits = compressed_bits,
            compression_ratio = compression_ratio
        )
        db.session.add(new_file)
        db.session.commit()

        return jsonify({
            "message": "Compression successful",
            "file_id": new_file.id,
            "filename": filename,
            "rows": rows,
            "cols": cols,
            "m": m,
            "original_size_bits": original_bits,
            "compressed_size_bits": compressed_bits,
            "compression_ratio": compression_ratio
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/decompress/<int:file_id>", methods=["GET"])
@token_required
def decompress_file(current_user, file_id):
    file_entry = CompressedFile.query.filter_by(id=file_id, uploaded_by=current_user.id).first()
    if not file_entry:
        return jsonify({"error": "File not found or access denied"}), 404

    try:
        matrix = golomb_decode_bytes(file_entry.data, file_entry.m, file_entry.rows, file_entry.cols)

        # Load file_metadata
        meta = json.loads(file_entry.file_metadata) if file_entry.file_metadata else {}
        columns = meta.get("columns")
        has_header = meta.get("has_header", False)
        delimiter = meta.get("delimiter", ",")
        original_ext = meta.get("original_ext", ".csv")

        # Build output filename
        base = os.path.splitext(file_entry.filename)[0]
        suffix = "decompressed.csv" if original_ext == ".csv" else "decompressed.npy"
        output_name = f"{base}_{file_id}_{int(datetime.utcnow().timestamp())}_{suffix}"
        # output_path = os.path.join(UPLOAD_FOLDER, output_name)

        # # Save appropriately
        # if original_ext == ".csv":
        #     df = pd.DataFrame(matrix, columns=columns)
        #     df.to_csv(output_path, index=False, sep=delimiter, header=has_header)
        #     mimetype = "text/csv"
        # else:
        #     np.save(output_path, matrix)
        #     mimetype = "application/octet-stream"
        #     output_name = output_name.replace(".csv", ".npy")

        # return send_file(
        #     output_path,
        #     mimetype=mimetype,
        #     as_attachment=True,
        #     download_name=os.path.basename(output_name)
        # )
        output_buffer = io.BytesIO()

        if original_ext == ".csv":
            df = pd.DataFrame(matrix, columns=columns)
            df.to_csv(output_buffer, index=False, sep=delimiter, header=has_header)
            output_buffer.seek(0)
            mimetype = "text/csv"
            download_name = f"{base}_{file_id}_{suffix.replace('.csv', '.csv')}"
        else:
            np.save(output_buffer, matrix)
            output_buffer.seek(0)
            mimetype = "application/octet-stream"
            download_name = f"{base}_{file_id}_{suffix}"

        return send_file(
            output_buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list-files', methods=['GET'])
@token_required
def list_files(current_user):
    """List all compressed files uploaded by the current user."""
    files = CompressedFile.query.filter_by(uploaded_by=current_user.id).all()
    result = []
    for f in files:
        result.append({
            'id': f.id,
            'filename': f.filename,
            'm': f.m,
            'rows': f.rows,
            'cols': f.cols,
            'size_bytes': len(f.data),
            'created_at': f.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'original_size_bits': f.original_size_bits,
            'compressed_size_bits': f.compressed_size_bits,
            'compression_ratio': f.compression_ratio,
        })
    return jsonify({'files': result})


# ================================================================
#  MAIN
# ================================================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
