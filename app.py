import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from indobert import SentimentAnalyzer
from flask import jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import os
import json
from flask import Flask, request, render_template, jsonify
import pickle
import torch
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims


app = Flask(__name__)

# Konfigurasi koneksi database MySQL
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''  # Sesuaikan dengan password database Anda
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_DB'] = 'data2'
app.config['SECRET_KEY'] = 'your_secret_key'

# Inisialisasi MySQL
mysql = MySQL(app)


# Inisialisasi SentimentAnalyzer dengan model IndoBERT yang sesuai
model_indobert = 'model'  # Ganti dengan path model yang benar jika menggunakan model lokal
analyzer = SentimentAnalyzer(model_indobert)

# Fungsi untuk mengambil data ulasan dari database
def get_reviews_from_db():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM reviews")
    reviews = cur.fetchall()
    return reviews

# Fungsi untuk mengambil data users dari database
def get_users_from_db():
    cur = mysql.connection.cursor()  # Membuka koneksi ke database
    cur.execute("SELECT id, username, email, dob FROM users")  # Query untuk mengambil data
    users = cur.fetchall()  # Mengambil semua hasil query
    cur.close()  # Menutup koneksi
    return users

# Initialize LLM
def initialize_llm(groq_api_key):
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    return llm

# Initialize embeddings
def initialize_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    return embeddings

# Create retrieval-augmented generation chain
def create_rag_chain(retriever, llm):
    system_prompt = (
        "Anda adalah asisten untuk tugas menjawab pertanyaan yang bernama gold. "
        "Gunakan konteks yang diambil untuk menjawab "
        "Menjawab menggunakan bahasa indonesia "
        "Jika Anda tidak ada jawaban pada konteks, katakan saja saya tidak tahu dan berikan jawaban yang sesuai "
        ". Gunakan maksimal empat kalimat dan pertahankan "
        "jawaban singkat.\n\n"
        "{context}"
    )
    retrieval_qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        | llm
        | StrOutputParser()
    )
    return retrieval_qa_chain

# Save model and vectorstore
def save_model(vectorstore, embeddings, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    vectorstore_path = os.path.join(save_dir, "vectorstore.pkl")
    with open(vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)
    embeddings_path = os.path.join(save_dir, "embeddings.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)

# Run chatbot logic
def run_chatbot(pdf_path, groq_api_key, save_dir="D:\\Semester 5\\web\\capstone\\model3"):
    llm = initialize_llm(groq_api_key)
    embeddings = initialize_embeddings()
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    save_model(vectorstore, embeddings, save_dir)
    rag_chain = create_rag_chain(retriever, llm)
    return rag_chain

# Define route for chatbot interaction
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    if question.lower() == 'keluar':
        return jsonify({"response": "Terima kasih! Sampai jumpa."})
    
    try:
        # Run the chatbot and get a response
        pdf_path = 'D:\\Semester 5\\web\\capstone\\static\\datachatbot.pdf'  # Specify the PDF file path
        groq_api_key = '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'  # Specify your Groq API key
        rag_chain = run_chatbot(pdf_path, groq_api_key)
        response = rag_chain.invoke(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Terjadi kesalahan: {e}"})


# Set folder for image uploads
app.config['UPLOAD_FOLDER'] = './images'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Check if upload folder exists, create it if not
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pre-trained model
model = load_model('D:\\Semester 5\\web-capstone\\capstone\\modell\\model_2.h5')

# Food information mapping
food_info_list = [
    {
        "nama": "durian",
        "kategori": "Tidak Aman",
        "alasan": "Durian memiliki kandungan gula tinggi dan sifat panas yang berpotensi memicu tekanan darah tinggi pada ibu hamil.",
        "rekomendasi": "Hindari durian, terutama jika memiliki riwayat tekanan darah tinggi atau diabetes gestasional."
    },
    {
        "nama": "nanas",
        "kategori": "Perlu Hati-Hati",
        "alasan": "Nanas mengandung bromelain yang dapat memicu kontraksi jika dikonsumsi dalam jumlah banyak, terutama pada trimester awal.",
        "rekomendasi": "Konsumsi nanas dalam jumlah kecil dan hindari jika memiliki risiko keguguran."
    },
    {
        "nama": "steak",
        "kategori": "Tidak Aman",
        "alasan": "Steak yang tidak matang sempurna berpotensi mengandung bakteri seperti E. coli atau parasit yang berbahaya bagi ibu hamil.",
        "rekomendasi": "Hindari steak yang tidak well-done. Ganti dengan daging matang lainnya sebagai sumber protein."
    },
    {
        "nama": "sushi",
        "kategori": "Tidak Aman",
        "alasan": "Sushi dengan ikan mentah memiliki risiko tinggi terhadap infeksi parasit dan bakteri seperti Listeria.",
        "rekomendasi": "Hindari sushi dengan ikan mentah. Pilih sushi dengan bahan matang seperti sayuran atau telur."
    },
    {
        "nama": "telur setengah matang",
        "kategori": "Tidak Aman",
        "alasan": "Telur setengah matang berisiko mengandung bakteri Salmonella yang dapat menyebabkan infeksi.",
        "rekomendasi": "Pilih telur yang dimasak matang sempurna sebagai alternatif."
    },
]

# Class names matching the model's training
class_names = ['durian', 'nanas', 'steak', 'sushi', 'telur setengah matang']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to get food info from the list
def get_food_info(food_name):
    for food in food_info_list:
        if food['nama'].lower() == food_name.lower():
            return food
    return {
        'kategori': 'Tidak Diketahui',
        'alasan': 'Informasi tidak tersedia.',
        'rekomendasi': 'Tidak ada rekomendasi.'
    }

# Function to get the prediction and corresponding food information
def get_output(img_path):
    try:
        # Load and preprocess the image
        loaded_img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(loaded_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Expanding dimensions for batch size 1

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_food = class_names[predicted_class]

        # Get food details (category, reason, recommendation)
        food_details = get_food_info(predicted_food)

        return {
            'food': predicted_food,
            'kategori': food_details['kategori'],
            'alasan': food_details['alasan'],
            'rekomendasi': food_details['rekomendasi']
        }
    except Exception as e:
        return {'error': f"Error processing image: {str(e)}"}

# Route for detecting food
@app.route('/deteksi_image')
def deteksi_image():
    return render_template('deteksi_image.html')

# Route for predicting based on uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    error = None
    uploaded_image = None
    try:
        # Handle uploaded file
        if 'image' not in request.files:
            error = 'Tidak ada file yang diunggah.'
            return render_template('deteksi_image.html', error=error)

        image = request.files['image']

        # Check if a file was selected
        if image.filename == '':
            error = 'Tidak ada file yang dipilih.'
            return render_template('deteksi_image.html', error=error)

        # Validate file type
        if not allowed_file(image.filename):
            error = 'Tipe file tidak valid. Hanya JPG, JPEG, dan PNG yang diperbolehkan.'
            return render_template('deteksi_image.html', error=error)

        # Save the image to the upload folder
        uploaded_image = image.filename
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image)
        image.save(img_path)

        # Get prediction
        prediction = get_output(img_path)

    except Exception as e:
        error = str(e)

    return render_template('deteksi_image.html', prediction=prediction, error=error, uploaded_image=uploaded_image)

# Route to display uploaded image
@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




# Route untuk halaman admin
@app.route('/data_user')
def data_user():
    users = get_users_from_db()  # Memanggil fungsi untuk mengambil data
    return render_template('data_user.html', users=users)



# Route untuk ulasan
@app.route('/ulasan')
def ulasan():
    reviews = session.get('reviews', [])
    return render_template('ulasan.html', reviews=reviews)

@app.route('/sentimen')
def sentimen():
    # Mengambil data ulasan dari database
    reviews = get_reviews_from_db()  # Menggunakan fungsi untuk mengambil ulasan dari database
    sentiment_results = []

    # Proses prediksi untuk setiap ulasan
    for review in reviews:
        review_text = review[1]  # Menyesuaikan indeks jika kolom 'text' ada di urutan kedua
        predicted_class, probabilities = analyzer.predict_sentiment(review_text)
        sentiment = "Positif" if predicted_class == 1 else "Negatif"  # Tentukan kategori sentimen
        sentiment_results.append({
            "text": review_text,
            "sentiment": sentiment
        })

    return render_template('prediksi_ulasan.html', sentiment_results=sentiment_results)

# Route untuk menambahkan ulasan
@app.route('/add_review', methods=['POST'])
def add_review():
    data = request.json
    review_text = data['text']

    # Menyimpan ulasan ke database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO reviews (text) VALUES (%s)", (review_text,))
    mysql.connection.commit()

    # Mengirim respons
    return jsonify({"text": review_text})

# Halaman Home
@app.route('/')
def home():
    return render_template('login.html')

# Konfigurasi akun admin
ADMIN_CREDENTIALS = {
    "username": "admin",
    "email": "admin@gmail.com",
    "password": "admin123"  # Ganti password dengan hash jika ingin lebih aman
}

# Halaman Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        dob = request.form['dob']

        if password != confirm_password:
            flash("Password and Confirm Password do not match!", 'danger')
            return redirect(url_for('register'))
        
        # Periksa apakah email sudah terdaftar
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_email = cursor.fetchone()
        if existing_email:
            flash("Email already registered!", 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Menyimpan data pengguna ke database MySQL
        cursor.execute('''INSERT INTO users (username, email, password, dob)
                          VALUES (%s, %s, %s, %s)''', (username, email, hashed_password, dob))
        mysql.connection.commit()
        cursor.close()

        flash("User Registered Successfully!", 'success')
        return redirect(url_for('home'))

    return render_template('register.html')

# Halaman Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Ambil input dari form login
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Cek apakah login sebagai admin
        if username == ADMIN_CREDENTIALS['username'] and email == ADMIN_CREDENTIALS['email'] and password == ADMIN_CREDENTIALS['password']:
            session['username'] = username  # Simpan username di session
            session['is_admin'] = True  # Tandai sebagai admin
            flash("Login successful as Admin!", 'success')
            return redirect(url_for('admin_dashboard'))

        # Cek apakah user ada di database
        cursor = mysql.connection.cursor()
        cursor.execute('''SELECT username, email, password, dob FROM users WHERE username = %s AND email = %s''', (username, email))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user[2], password):  # user[2] adalah password yang di-hash
            session['username'] = user[0]  # Simpan username di session
            session['dob'] = user[3]  # Simpan tanggal lahir di session
            session['is_admin'] = False  # Tandai sebagai user biasa

            flash("Login successful!", 'success')
            return redirect(url_for('profile'))
        else:
            flash("Invalid username, email, or password!", 'danger')

    return render_template('login.html')

@app.route('/admin')
def admin_dashboard():
    if 'is_admin' in session and session['is_admin']:
        return render_template('admin_dashboard.html', username=session['username'])
    flash("Access Denied! Admins only.", 'danger')
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Ambil data pengguna dari session
    username = session['username']
    dob = session['dob']
    
    # Perbaiki parsing tanggal agar sesuai dengan format yang benar
    dob_date = datetime.strptime(dob, '%a, %d %b %Y %H:%M:%S GMT')
    
    # Hitung umur berdasarkan tanggal lahir
    age = (datetime.now() - dob_date).days // 365
    
    return render_template('profile.html', username=username, age=age)

# Halaman Home
@app.route('/user_dashboard')
def user_dashboard():
    return render_template('user_dashboard.html')

# Data artikel statis dengan URL baru
articles = [
    {
        "id": 2,
        "url": "https://bunda.co.id/artikel/kesehatan/kehamilan/penting-16-rekomendasi-cemilan-sehat-ibu-hamil/",
        "title": "16 Rekomendasi Cemilan Sehat untuk Ibu Hamil - Bunda",
    },
    {
        "id": 3,
        "url": "https://www.halodoc.com/kesehatan/kehamilan?srsltid=AfmBOoqMEg4XhFwuTNAUdDBiy7589Q0atWCSvTnXuOn30HmS0smcQEjZ",
        "title": "Kehamilan - Halodoc",
    },
    {
        "id": 4,
        "url": "https://www.alodokter.com/lawan-9-masalah-tidur-ibu-hamil-dengan-cara-cara-ini",
        "title": "9 Masalah Tidur Ibu Hamil dan Cara Mengatasinya - Alodokter",
    },
    {
        "id": 5,
        "url": "https://www.alodokter.com/5-makanan-yang-harus-dihindari-selama-kehamilan",
        "title": "5 Makanan yang Harus Dihindari Selama Kehamilan - Alodokter",
    },
    {
        "id": 6,
        "url": "https://www.alodokter.com/waspadai-risiko-kelebihan-asam-folat-pada-ibu-hamil",
        "title": "Waspadai Risiko Kelebihan Asam Folat pada Ibu Hamil - Alodokter",
    },
    {
        "id": 7,
        "url": "https://bunda.co.id/artikel/kesehatan/kehamilan/12-gerakan-senam-hamil-di-rumah-yang-mudah-dipraktikkan/",
        "title": "12 Gerakan Senam Hamil di Rumah yang Mudah Dipraktikkan - Bunda",
    },
    {
        "id": 8,
        "url": "https://bunda.co.id/artikel/kesehatan/kehamilan/percepat-pemulihan-pasca-persalinan-dengan-teknik-eras-apa-itu/",
        "title": "Percepat Pemulihan Pasca Persalinan dengan Teknik ERAS - Bunda",
    },
    {
        "id": 9,
        "url": "https://www.prenagen.com/id/posisi-tidur-untuk-plasenta-previa",
        "title": "Posisi Tidur untuk Plasenta Previa - Prenagen",
    },
    {
        "id": 10,
        "url": "https://www.prenagen.com/id/penyebab-down-syndrome",
        "title": "Penyebab Down Syndrome - Prenagen",
    },
    {
        "id": 11,
        "url": "https://www.prenagen.com/id/gerakan-janin-yang-tidak-normal",
        "title": "Gerakan Janin yang Tidak Normal - Prenagen",
    },
    {
        "id": 12,
        "url": "https://www.prenagen.com/id/makanan-agar-cepat-kontraksi",
        "title": "Makanan agar Cepat Kontraksi - Prenagen",
    },
    {
        "id": 13,
        "url": "https://www.prenagen.com/id/protein-untuk-mencegah-stunting-sejak-kehamilan",
        "title": "Protein untuk Mencegah Stunting Sejak Kehamilan - Prenagen",
    },
    {
        "id": 14,
        "url": "https://www.prenagen.com/id/makanan-untuk-mengurangi-kaki-bengkak-pada-ibu-hamil",
        "title": "Makanan untuk Mengurangi Kaki Bengkak pada Ibu Hamil - Prenagen",
    },
    {
        "id": 16,
        "url": "https://www.prenagen.com/id/posisi-tidur-agar-bayi-tidak-sungsang",
        "title": "Posisi Tidur Agar Bayi Tidak Sungsang - Prenagen",
    }
    # Tambahkan artikel lainnya sesuai kebutuhan
]

# Fungsi untuk scraping artikel dari sebuah URL
def scrape_article(url):
    """Scrape judul, isi artikel, dan gambar dari sebuah URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Menangkap judul artikel
            title = soup.find('h1').text.strip() if soup.find('h1') else "Judul tidak ditemukan"
            
            # Menangkap isi artikel (p, section) dan menyaring informasi yang tidak relevan
            paragraphs = soup.find_all('p')
            content = [p.text.strip() for p in paragraphs if p.text.strip() and not is_irrelevant_content(p.text)]
            
            # Menangkap gambar yang relevan
            images = [img['src'] for img in soup.find_all('img') if img.get('src') and is_relevant_image(img)]
            
            return {"title": title, "content": content, "images": images}
        
        return {"title": "Artikel tidak tersedia", "content": [], "images": []}
    
    except Exception as e:
        return {"title": "Error mengambil artikel", "content": [str(e)], "images": []}

def is_irrelevant_content(text):
    """Menentukan apakah teks ini tidak relevan, seperti nama dokter, tanggal, dll."""
    irrelevant_keywords = ["dokter", "tanggal", "hubungi", "jam", "penulis"]
    return any(keyword.lower() in text.lower() for keyword in irrelevant_keywords)

def is_relevant_image(img):
    """Menentukan apakah gambar tersebut relevan untuk artikel (misalnya, gambar produk atau ilustrasi)."""
    relevant_keywords = ["gambar", "ilustrasi", "produk"]
    return any(keyword.lower() in img.get('alt', '').lower() for keyword in relevant_keywords)

# Contoh penggunaan fungsi scrape_article untuk mengambil artikel pertama
article_data = scrape_article(articles[0]['url'])
print(article_data)

# Rute untuk menampilkan daftar artikel
@app.route('/artikel')
def artikel():
    return render_template('artikel.html', articles=articles)

# Rute untuk menampilkan detail artikel
@app.route('/artikel/<int:article_id>')
def article(article_id):
    article_data = next((article for article in articles if article['id'] == article_id), None)
    if article_data:
        scraped_data = scrape_article(article_data["url"])
        return render_template('healthtips.html', article=scraped_data)
    return "Artikel tidak ditemukan", 404

#About
@app.route('/about')
def about():
    features = [
        {"judul": "Registrasi", "deskripsi": "Mendaftar akun untuk memulai pengalaman yang dipersonalisasi.", "ikon": "fas fa-user"},
        {"judul": "Login", "deskripsi": "Masuk ke aplikasi dengan akun terdaftar Anda.", "ikon": "fas fa-sign-in-alt"},
        {"judul": "Profil", "deskripsi": "Kelola informasi pribadi Anda di satu tempat.", "ikon": "fas fa-user-circle"},
        {"judul": "Home", "deskripsi": "Temukan informasi penting langsung dari halaman utama.", "ikon": "fas fa-home"},
        {"judul": "Checklist", "deskripsi": "Checklist persiapan kehamilan dan kelahiran untuk mempermudah Anda.", "ikon": "fas fa-check-circle"},
        {"judul": "Jurnal", "deskripsi": "Catat perkembangan mingguan Anda di jurnal digital.", "ikon": "fas fa-book"},
        {"judul": "Kalender", "deskripsi": "Atur jadwal penting Anda dengan fitur kalender.", "ikon": "fas fa-calendar-alt"},
        {"judul": "Prediksi HPL", "deskripsi": "Menghitung perkiraan Hari Perkiraan Lahir (HPL).", "ikon": "fas fa-calendar-check"},
        {"judul": "Deteksi Makanan", "deskripsi": "Mendeteksi makanan yang aman dan tidak aman untuk ibu hamil.", "ikon": "fas fa-utensils"},
        {"judul": "Chatbot", "deskripsi": "Konsultasi cepat dengan fitur chatbot.", "ikon": "fas fa-comment-dots"},
        {"judul": "Jurnal Mingguan", "deskripsi": "Memberikan informasi mingguan tentang perkembangan ibu hamil.", "ikon": "fas fa-bookmark"},
        {"judul": "Rekomendasi Produk", "deskripsi": "Memberikan saran produk yang aman untuk ibu dan bayi.", "ikon": "fas fa-box"}
    ]
    return render_template('tentang_kami.html', features=features)

# Fungsi menghitung HPL
def hitung_hpl(tanggal_haid):
    try:
        date = datetime.strptime(tanggal_haid, "%Y-%m-%d")
        hpl = date + timedelta(days=280)  # Kehamilan rata-rata 40 minggu
        return hpl.strftime("%d-%m-%Y")
    except ValueError:
        return None# Format tanggal tidak valid
# Rute untuk Cek HPL
@app.route('/cek-hpl', methods=["GET", "POST"])
def cek_hpl():
    hpl = None
    error = None
    if request.method == "POST":
        tanggal_haid = request.form.get("tanggal_haid")
        hpl = hitung_hpl(tanggal_haid)
        if not hpl:
            error = "Format tanggal tidak valid. Gunakan format YYYY-MM-DD."
    return render_template('cek_hpl.html', hpl=hpl, error=error)

# API untuk menghitung HPL
@app.route('/api/hpl', methods=["POST"])
def api_hpl():
    if not request.is_json:
        return jsonify({"error": "Harap kirimkan data dalam format JSON"}), 415

    data = request.get_json()
    tanggal_haid = data.get('tanggal_haid')
    if not tanggal_haid:
        return jsonify({"error": "Tanggal haid tidak diberikan"}), 400

    hpl = hitung_hpl(tanggal_haid)
    return jsonify({"hpl": hpl}) if hpl else jsonify({"error": "Tanggal tidak valid"}), 400


# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", 'success')
    return redirect(url_for('home'))

# Menjalankan Aplikasi
if __name__ == "__main__":
    app.run(debug=True)
