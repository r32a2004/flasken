import os
import io
import time
import re
import cv2
import xlwt
import pymysql
import numpy as np
import pandas as pd
import requests
import PyPDF2
import mysql.connector
from PIL import Image
from calendar import monthrange
from datetime import date, datetime
from openpyxl import Workbook
from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# ==============================================================================
# INISIALISASI APLIKASI FLASK DAN KONFIGURASI
# ==============================================================================
app = Flask(__name__,            # pastikan nama variabel konsisten
            template_folder='templates',
            static_folder='static')
CORS(app)  # Diambil dari nabil.py untuk mengizinkan request API
app.secret_key = 'bebasapasaja'  # Diambil dari app.py

# ==============================================================================
# VARIABEL GLOBAL & KONSTANTA DARI KEDUA FILE
# ==============================================================================

# --- Dari app.py (Sistem Absensi) ---
cnt = 0
pause_cnt = 0
justscanned = False
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.venv/dataset')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Dari nabil.py (NLP & API) ---
MODEL_NAME = "abdmuffid/fine-tuned-indo-sentiment-3-class"
API_KEY = "YOUR_API_KEY"
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "deepseek-r1-distill-llama-70b"
pdf_text_global = ""

# ==============================================================================
# KONEKSI DATABASE & PEMUATAN MODEL
# ==============================================================================

# --- Koneksi Database dari app.py ---
mydb = mysql.connector.connect(
    host="srv590.hstgr.io",
    user="u829376119_phhkm",
    passwd="D3s@c1putriUmBB",
    database="u829376119_phhkm"
)
mycursor = mydb.cursor()

# --- Pemuatan Model NLP dari nabil.py ---
def download_model_once(model_name):
    """Downloads the Hugging Face model and tokenizer if not already cached."""
    try:
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Model dan tokenizer sudah ada di cache atau berhasil di-download.")
    except Exception as e:
        print(f"Gagal download model/tokenizer: {e}")

download_model_once(MODEL_NAME)
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME
)

# ==============================================================================
# FUNGSI HELPER DARI KEDUA FILE
# ==============================================================================

# --- Helper dari app.py ---
def resource_path(*relative_path):
    """Return absolute path to resource, relative to this file."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, *relative_path)

# --- Helper dari nabil.py ---
def bersihkan_teks(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:?!-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_think_tags(text):
    return re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()

def query_model(chat_history):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": chat_history
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if resp.status_code == 200:
        result = resp.json()
        raw_content = result['choices'][0]['message']['content']
        clean_content = remove_think_tags(raw_content)
        return clean_content
    else:
        try:
            err = resp.json()
        except:
            err = resp.text
        return f"Terjadi kesalahan: {err}"


# ==============================================================================
# BAGIAN I: FUNGSI DAN RUTE DARI SISTEM ABSENSI (app.py)
# ==============================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        resource_path('../resources/haarcascade_frontalface_default.xml')
    )
    eye_classifier = cv2.CascadeClassifier(
        resource_path('../resources/haarcascade_eye.xml')
    )

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray)
            if len(eyes) >= 1:
                cropped_face = img[y:y + h, x:x + w]
                return cropped_face
        return None

    cap = cv2.VideoCapture(0)
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = os.path.join(UPLOAD_FOLDER, f"{nbr}.{img_id}.jpg")
            file_name_path2 = f"{nbr}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`, `img_path`) VALUES
                                ('{}', '{}', '{}')""".format(img_id, nbr, file_name_path2))
            mydb.commit()
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
    cap.release()
    cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = UPLOAD_FOLDER
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write(resource_path('../.venv/classifier.xml'))
    return redirect('/petugas')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():
    faceCascade = cv2.CascadeClassifier(resource_path('../resources/haarcascade_frontalface_default.xml'))
    eyeCascade = cv2.CascadeClassifier(resource_path('../resources/haarcascade_eye.xml'))
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read(resource_path('../.venv/classifier.xml'))

    if faceCascade.empty() or eyeCascade.empty():
        raise IOError("Cascade file not loaded. Cek path dan file XML.")

    wCam, hCam = 400, 400

    def draw_boundary(img, classifier, eye_classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        global justscanned, pause_cnt, cnt
        pause_cnt += 1
        coords = []

        for (x, y, w, h) in features:
            roi_gray = gray_image[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            if len(eyes) < 1:
                continue

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                cnt += 1
                if cnt > 30: cnt = 30
                
                n = (100 / 30) * cnt
                w_filled = (cnt / 30) * w
                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

                mycursor.execute("select a.img_person, b.prs_name from img_dataset a left join prs_mstr b on a.img_person = b.prs_nbr where img_id = " + str(id))
                row = mycursor.fetchone()
                if row is None:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    continue

                pnbr, pname = row[0], row[1] if row[1] is not None else ""
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                jam_sekarang = now.strftime("%H:%M:%S")
                masuk_start, masuk_end = "06:00:00", "08:00:00"
                pulang_start, pulang_end = "13:00:00", "23:59:59"

                if masuk_start <= current_time <= masuk_end:
                    statusabsen = "Absen Masuk"
                    cv2.putText(img, pname + '-' + statusabsen, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 221, 34), 2, cv2.LINE_AA)
                elif current_time > masuk_end and current_time < pulang_start:
                    statusabsen = "Absen Masuk (Terlambat)"
                    cv2.putText(img, pname + '-' + statusabsen, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                elif pulang_start <= current_time <= pulang_end or current_time < masuk_start:
                    statusabsen = "Absen Pulang"
                    cv2.putText(img, pname + '-' + statusabsen, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, 'Di luar jam absensi', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    return coords

                if int(cnt) == 30:
                    cnt = 0
                    cursor = mydb.cursor()
                    cursor.execute('SELECT * FROM accs_hist WHERE accs_prsn=%s and accs_date=%s', (pnbr, date.today()))
                    absen = cursor.fetchone()
                    if absen is None:
                        if masuk_start <= current_time <= masuk_end:
                            cursor.execute("INSERT INTO accs_hist (accs_date, accs_prsn, masuk, status) VALUES (%s, %s, %s, %s)", (date.today(), pnbr, jam_sekarang, "Hadir"))
                        elif current_time > masuk_end and current_time < pulang_start:
                            cursor.execute("INSERT INTO accs_hist (accs_date, accs_prsn, masuk, status) VALUES (%s, %s, %s, %s)", (date.today(), pnbr, jam_sekarang, "Terlambat"))
                        elif pulang_start <= current_time <= pulang_end or current_time < masuk_start:
                            cursor.execute("INSERT INTO accs_hist (accs_date, accs_prsn, keluar) VALUES (%s, %s, %s)", (date.today(), pnbr, jam_sekarang))
                        mydb.commit()
                    else:
                        if pulang_start <= current_time <= pulang_end or current_time < masuk_start:
                            if absen[3] is None:
                                cursor.execute("UPDATE accs_hist SET keluar=%s WHERE accs_prsn=%s and accs_date=%s", (jam_sekarang, pnbr, date.today()))
                                mydb.commit()
                        elif masuk_start <= current_time <= pulang_start:
                            if absen[2] is None:
                                if masuk_start <= current_time <= masuk_end:
                                    cursor.execute("UPDATE accs_hist SET masuk=%s, status=%s WHERE accs_prsn=%s and accs_date=%s", (jam_sekarang, "Hadir", pnbr, date.today()))
                                elif current_time > masuk_end:
                                    cursor.execute("UPDATE accs_hist SET masuk=%s, status=%s WHERE accs_prsn=%s and accs_date=%s", (jam_sekarang, "Terlambat", pnbr, date.today()))
                                mydb.commit()
                    time.sleep(1)
                    justscanned = True
                    pause_cnt = 0
            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, '', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                if pause_cnt > 80:
                    justscanned = False
            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade, eyeCascade):
        coords = draw_boundary(img, faceCascade, eyeCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade, eyeCascade)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/petugas')
def petugas():
    mycursor.execute("select prs_nbr, prs_name from prs_mstr")
    data = mycursor.fetchall()
    return render_template('index.html', data=data)

@app.route('/admin')
def admin():
    mycursor.execute("select id_user, name, email, role from users")
    data = mycursor.fetchall()
    return render_template('admin.html', data=data)

@app.route('/editadmin/<_id_admin>')
def editadmin(_id_admin):
    if 'loggedin' not in session or session.get('level') != 'Admin':
        return redirect(url_for('login'))
    cursor = mydb.cursor()
    sql = "SELECT id, username, email, level, date_input FROM tb_users WHERE id = %s"
    params = (_id_admin,)
    cursor.execute(sql, params)
    row = cursor.fetchone()
    return render_template('editregistrasi.html', data=row)

@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    mycursor.execute("SELECT id_user, name, email, role FROM users WHERE role='2' AND id_user NOT IN (SELECT user_id FROM prs_mstr)")
    karyawan_baru = mycursor.fetchall()
    return render_template('addprsn.html', newnbr=int(nbr), karyawan_baru=karyawan_baru)

@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    user_id = request.form.get('user_id')
    prsname = request.form.get('txtname')
    mycursor.execute("INSERT INTO `prs_mstr` (`prs_nbr`, `user_id`, `prs_name`) VALUES (%s, %s, %s)", (prsnbr, user_id, prsname))
    mydb.commit()
    return redirect(url_for('vfdataset_page', prs=prsnbr))

@app.route('/edit/<_prs_nbr>')
def editprsn(_prs_nbr):
    if 'loggedin' not in session: return redirect(url_for('login'))
    cursor = mydb.cursor()
    sql = "SELECT * FROM prs_mstr WHERE prs_nbr = %s"
    data = (_prs_nbr,)
    cursor.execute(sql, data)
    row = cursor.fetchone()
    return render_template('editprsn.html', data=row)

@app.route('/editprsn_submit', methods=['POST'])
def editprsn_submit():
    if 'loggedin' not in session: return redirect(url_for('login'))
    _prsnbr = request.form.get('txtnbr')
    _prsname = request.form.get('txtname')
    _prsskill = request.form.get('optskill')
    mycursor = mydb.cursor()
    sql = "UPDATE `prs_mstr` set prs_name= '" + _prsname + "', prs_skill= '" + _prsskill + "' where prs_nbr = '" + _prsnbr + "'"
    mycursor.execute(sql)
    mydb.commit()
    return redirect('/petugas')

@app.route('/delete/<_prs_nbr>')
def deleteimg(_prs_nbr):
    mycursor.execute("SELECT img_id FROM img_dataset WHERE img_person = %s LIMIT 1", (_prs_nbr,))
    data = mycursor.fetchone()
    if not data: return "Data gambar tidak ditemukan.", 404

    img_id = int(data[0])
    for i in range(img_id, img_id + 100):
        filepath = os.path.join(UPLOAD_FOLDER, f"{_prs_nbr}.{i}.jpg")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Hapus: {filepath}")
        else:
            print(f"File tidak ditemukan: {filepath}")

    mycursor.execute("DELETE FROM img_dataset WHERE img_person = %s", (_prs_nbr,))
    mydb.commit()
    mycursor.execute("DELETE FROM prs_mstr WHERE prs_nbr = %s", (_prs_nbr,))
    mydb.commit()
    mycursor.execute("DELETE FROM accs_hist WHERE accs_prsn = %s", (_prs_nbr,))
    mydb.commit()
    return redirect('/train_classifier/' + _prs_nbr)

@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)

@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/riwayat_absensi')
def riwayat_absensi():
    mydb_local = mysql.connector.connect(host="srv590.hstgr.io", user="u829376119_phhkm", passwd="D3s@c1putriUmBB", database="u829376119_phhkm")
    cur = mydb_local.cursor()
    start, end, prs_nbr = request.args.get('start'), request.args.get('end'), request.args.get('prs_nbr')
    sql = "SELECT a.*, b.prs_name FROM accs_hist a LEFT JOIN prs_mstr b ON a.accs_prsn=b.prs_nbr"
    params, people = [], []
    if session.get('level') == 'Karyawan':
        username = session.get('username')
        cur.execute("SELECT id FROM tb_users WHERE username=%s", (username,))
        user_row = cur.fetchone()
        if user_row:
            sql += " WHERE b.user_id = %s"; params.append(user_row[0])
        else:
            mydb_local.close(); return "User tidak ditemukan", 404
    else:
        cur.execute("SELECT prs_nbr, prs_name FROM prs_mstr"); people = cur.fetchall()
        if prs_nbr:
            sql += " WHERE a.accs_prsn = %s"; params.append(prs_nbr)
    if start: sql += (" AND " if params else " WHERE ") + "a.accs_date >= %s"; params.append(start)
    if end: sql += (" AND " if params else " WHERE ") + "a.accs_date <= %s"; params.append(end)
    sql += " ORDER BY a.accs_date DESC, a.accs_added DESC"

    cur.execute(sql, tuple(params))
    rows = [list(l) for l in cur.fetchall()]
    for l in rows:
        try:
            if isinstance(l[4], str): l[4] = datetime.strptime(l[4], '%Y-%m-%d')
        except Exception: pass
    
    mydb_local.close()
    return render_template('riwayat_absensi.html', logs=rows, start=start, end=end, people=people, prs_nbr=prs_nbr)

@app.route('/riwayat_absensi/<int:id>/edit', methods=('GET', 'POST'))
def riwayat_edit(id):
    cur = mydb.cursor()
    if request.method == 'POST':
        keg = request.form['kegiatan']
        if session.get('level') in ['Petugas', 'Admin']:
            status = request.form['status']
            cur.execute("UPDATE accs_hist SET status=%s, kegiatan=%s WHERE accs_id=%s", (status, keg, id))
        else:
            cur.execute("UPDATE accs_hist SET kegiatan=%s WHERE accs_id=%s", (keg, id))
        mydb.commit()
        return redirect(url_for('riwayat_absensi'))

    cur.execute("SELECT * FROM accs_hist WHERE accs_id=%s", (id,))
    log = cur.fetchone()
    cur.execute("SELECT prs_nbr, prs_name FROM prs_mstr"); people = cur.fetchall()
    return "Form tambah absensi manual sedang dinonaktifkan.", 403

@app.route('/')
def fr_page():
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, a.accs_added from accs_hist a left join prs_mstr b on a.accs_prsn = b.prs_nbr where a.accs_date = curdate() order by 1 desc")
    data = mycursor.fetchall()
    return render_template('fr_page.html', data=data)

@app.route('/countTodayScan')
def countTodayScan():
    mydb_local = mysql.connector.connect(host="srv590.hstgr.io", user="u829376119_phhkm", passwd="D3s@c1putriUmBB", database="u829376119_phhkm")
    mycursor_local = mydb_local.cursor()
    mycursor_local.execute("select count(*) from accs_hist where accs_date = curdate()")
    row = mycursor_local.fetchone()
    rowcount = row[0]
    return jsonify({'rowcount': rowcount})

@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    mydb_local = mysql.connector.connect(host="srv590.hstgr.io", user="u829376119_phhkm", passwd="D3s@c1putriUmBB", database="u829376119_phhkm")
    mycursor_local = mydb_local.cursor()
    mycursor_local.execute("SELECT a.accs_id, a.accs_prsn, b.prs_name, a.status, DATE_FORMAT(a.accs_added, '%Y-%m-%d'), a.masuk, a.keluar FROM accs_hist a LEFT JOIN prs_mstr b ON a.accs_prsn = b.prs_nbr WHERE a.accs_date = curdate() ORDER BY 1 DESC")
    data = mycursor_local.fetchall()
    result = []
    for row in data:
        row = list(row)
        for idx in [5, 6]:
            if isinstance(row[idx], (datetime,)): row[idx] = row[idx].strftime('%H:%M:%S')
            elif row[idx] is not None: row[idx] = str(row[idx])
            else: row[idx] = None
        result.append(row)
    return jsonify(response=result)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('level', None)
    return redirect(url_for('login'))

@app.route('/karyawan')
def karyawan():
    username = session.get('username')
    cursor = mydb.cursor()
    cursor.execute("SELECT username, email, level, date_input FROM tb_users WHERE username=%s", (username,))
    user = cursor.fetchone()
    return render_template('karyawan.html', user=user)

@app.route('/cetak_riwayat_absensi')
def cetak_riwayat_absensi():
    username, level, prs_nbr = session.get('username'), session.get('level'), request.args.get('prs_nbr')
    today = datetime.today()
    if today.day >= 14:
        start_date = today.replace(day=14)
        end_date = (today.replace(month=today.month % 12 + 1, day=13)) if today.month != 12 else today.replace(year=today.year + 1, month=1, day=13)
    else:
        start_date = (today.replace(month=today.month - 1, day=14)) if today.month != 1 else today.replace(year=today.year - 1, month=12, day=14)
        end_date = today.replace(day=13)

    cur = mydb.cursor()
    cur.execute("SELECT id_user FROM users WHERE name=%s", (username,)); user_row = cur.fetchone()
    user_id = user_row[0] if user_row else None
    people, prs_name = [], None

    if level == 'Karyawan':
        cur.execute("SELECT prs_nbr, prs_name FROM prs_mstr WHERE user_id=%s", (user_id,))
        prsn = cur.fetchone()
        if not prsn: return "Data karyawan tidak ditemukan", 404
        prs_nbr, prs_name = prsn
        sql = "SELECT a.accs_date, a.masuk, a.keluar, a.kegiatan, a.status, a.accs_id FROM accs_hist a WHERE a.accs_prsn=%s AND a.accs_date BETWEEN %s AND %s ORDER BY a.accs_date"
        cur.execute(sql, (prs_nbr, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
    else:
        cur.execute("SELECT prs_nbr, prs_name FROM prs_mstr"); people = cur.fetchall()
        base_sql = "SELECT a.accs_date, a.masuk, a.keluar, a.kegiatan, a.status, a.accs_id, b.prs_name FROM accs_hist a LEFT JOIN prs_mstr b ON a.accs_prsn=b.prs_nbr"
        if prs_nbr:
            cur.execute("SELECT prs_name FROM prs_mstr WHERE prs_nbr=%s", (prs_nbr,)); row = cur.fetchone()
            prs_name = row[0] if row else None
            sql = f"{base_sql} WHERE a.accs_prsn=%s AND a.accs_date BETWEEN %s AND %s ORDER BY a.accs_date"
            cur.execute(sql, (prs_nbr, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        else:
            sql = f"{base_sql} WHERE a.accs_date BETWEEN %s AND %s ORDER BY a.accs_prsn, a.accs_date"
            cur.execute(sql, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
    
    logs = cur.fetchall()
    return render_template('cetak_riwayat_absensi.html', logs=logs, level=level, prs_name=prs_name, start_date=start_date, end_date=end_date, people=people, prs_nbr=prs_nbr)


# ==============================================================================
# BAGIAN II: FUNGSI DAN RUTE DARI NLP & API (nabil.py)
# ==============================================================================

@app.route('/sentimen', methods=['POST'])
def sentimen():
    data = request.json
    print("Data diterima:", data)
    ulasan = data.get('ulasan', '')
    if not ulasan: return jsonify({"error": "ulasan kosong"}), 400
    try:
        result = sentiment_analysis(ulasan)
        label, score = result[0]['label'], result[0]['score']
        return jsonify({"class": label, "score": score})
    except Exception as e:
        print("Error saat analisis:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/helo')
def hello():
    return jsonify({"message": "Hello from Flask!"})

@app.route('/terima-data', methods=['POST'])
def terima_data():
    ulasan_data = request.json.get('ulasan_data', [])
    umkm_data = request.json.get('umkmData', [])

    df_ulasan = pd.DataFrame(ulasan_data)
    df_umkm = pd.DataFrame(umkm_data)

    if 'id_umkm' not in df_umkm.columns:
        return jsonify({'error': 'Kolom id_umkm tidak ditemukan pada data UMKM'}), 400

    df_ulasan['ringkasan_umkm'] = df_ulasan['ringkasan_umkm'].apply(bersihkan_teks)
    df_umkm['ringkasan_umkm'] = df_umkm['ringkasan_umkm'].apply(bersihkan_teks)
    def gabungkan_kolom(row): return f"{row.get('nama_umkm','')} {row.get('ringkasan_umkm','')} {row.get('produk','')}"
    
    df_ulasan['teks_gabungan'] = df_ulasan.apply(gabungkan_kolom, axis=1)
    df_umkm['teks_gabungan'] = df_umkm.apply(gabungkan_kolom, axis=1)
    semua_teks = pd.concat([df_ulasan['teks_gabungan'], df_umkm['teks_gabungan']]).reset_index(drop=True)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(semua_teks)
    tfidf_ulasan = vectorizer.transform(df_ulasan['teks_gabungan'])
    tfidf_umkm = vectorizer.transform(df_umkm['teks_gabungan'])
    cosine_sim_matrix = cosine_similarity(tfidf_ulasan, tfidf_umkm)

    top_n = 5
    rekomendasi_list = []
    umkm_terpakai = set()

    for i in range(cosine_sim_matrix.shape[0]):
        sorted_indices = cosine_sim_matrix[i].argsort()[::-1]
        rekomendasi_umkm = []
        for idx in sorted_indices:
            id_umkm = df_umkm.iloc[idx]['id_umkm']
            if id_umkm not in umkm_terpakai:
                umkm_terpakai.add(id_umkm)
                rekomendasi_umkm.append({
                    'id_umkm': id_umkm,
                    'nama_umkm': df_umkm.iloc[idx]['nama_umkm'],
                    'ringkasan_umkm': df_umkm.iloc[idx]['ringkasan_umkm'],
                    'produk': df_umkm.iloc[idx]['produk']
                })
            if len(rekomendasi_umkm) == top_n: break
        
        rekomendasi_list.append({
            'ulasan': df_ulasan.iloc[i][['nama_umkm', 'ringkasan_umkm', 'produk']].to_dict(),
            'rekomendasi_umkm': rekomendasi_umkm
        })
    return jsonify({'status': 'diterima', 'jumlah_ulasan': len(ulasan_data), 'jumlah_umkm': len(umkm_data), 'rekomendasi': rekomendasi_list})

@app.route('/anomali', methods=['POST'])
def deteksi_anomali():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Data harus berupa list transaksi'}), 400

    df = pd.DataFrame(data)
    if 'kategori' in df.columns:
        df['tipe'] = df['kategori']
    else:
        return jsonify({'error': 'Kolom kategori tidak ditemukan'}), 400

    df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
    df = df.dropna(subset=['jumlah'])
    df["status"] = "✅ Aman"
    df.loc[df["jumlah"] < 10000, "status"] = "🧐 Perlu Diaudit"
    df.loc[df["jumlah"] > 10000000, "status"] = "⚠ Warning"

    df_pengeluaran = df[(df["jumlah"] >= 10000) & (df["jumlah"] <= 10000000)]
    for tipe in df_pengeluaran["tipe"].unique():
        group = df_pengeluaran[df_pengeluaran["tipe"] == tipe]
        if len(group) < 3:
            fallback_idx = group[group["jumlah"] < 100000].index
            df.loc[fallback_idx, "status"] = "🧐 Perlu Diaudit"
            continue
        Q1, Q3 = group["jumlah"].quantile(0.25), group["jumlah"].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_idx = group[(group["jumlah"] < lower) | (group["jumlah"] > upper)].index
        df.loc[outlier_idx, "status"] = "🧐 Perlu Diaudit"
    return jsonify(df.to_dict(orient="records"))

@app.route('/api/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_text_global
    if 'pdf_file' not in request.files:
        return jsonify({"error": "Tidak ada file PDF yang diupload"}), 400

    pdf_file = request.files['pdf_file']
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages = [page.extract_text() or '' for page in pdf_reader.pages]
        pdf_text_global = "\n".join(pages).strip()
        if not pdf_text_global: return jsonify({"error": "File PDF kosong atau tidak bisa dibaca"}), 400

        chat_history = [
            {"role": "system", "content": "Anda adalah asisten yang merangkum isi dokumen secara singkat dan jelas."},
            {"role": "user", "content": f"Tolong buat ringkasan dari teks berikut:\n{pdf_text_global}"}
        ]
        response_text = query_model(chat_history)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file PDF: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global pdf_text_global
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    if not prompt: return jsonify({"error": "Prompt tidak boleh kosong"}), 400

    chat_history = [{"role": "system", "content": "Sebagai admin yang berbahasa Indonesia, saya memiliki peran penting dalam mengelola chatbot Transparansi Dana Desa... (dan seterusnya)"}]
    if pdf_text_global:
        chat_history.append({"role": "system", "content": f"Berikut isi dokumen PDF yang sudah diupload:\n{pdf_text_global}"})
    chat_history.append({"role": "user", "content": prompt})
    response_text = query_model(chat_history)
    return jsonify({"response": response_text})


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
             port = int(os.environ.get("PORT", 5000))  # default 5000 utk dev
    app.run(host="0.0.0.0", port=port, debug=True)
