🛒 Shoplifting Detection using VideoMAE & Django
This project is a **web-based shoplifting detection system** built with **Django** and **PyTorch**.  
It uses a **pre-trained VideoMAE model** to analyze surveillance video clips and classify whether the person is a **shop lifter** or **non shop lifter**.  

🚀 Features
- 🎥 Upload video clips for inference.  
- 🤖 Deep learning model (VideoMAE) for action recognition.  
- ✅ Displays result with confidence score.  
- 🖥️ Modern Bootstrap-based UI with Navbar & Footer.  
- ⚡ Easy to run locally with Django.  

📂 Project Structure
shoplift_django/
│
├── ShopLifter/ # Main Django project
├── detector/ # App handling detection logic
├── models/ # Pre-trained model files
│ └── videomae-shoplifting/
│ ├── config.json
│ ├── inference_config.json
│ ├── pytorch_model.bin <-- Download & place model here
├── db.sqlite3
├── manage.py
└── README.md

🔧 Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/TasneemElgendy/shoplifting-detection.git
cd shoplifting-detection

2️⃣ Install Required Packages
Make sure you have Python (>=3.8) installed.
Then install Django and PyTorch manually:
pip install django
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

3️⃣ Download Pre-trained Model
Download the pre-trained VideoMAE model from Google Drive:
🔗 Download Model: https://drive.google.com/file/d/1OtqVNIz_9j-91Y3oCqq2kAZTwyE6xZRc/view?usp=sharing
Place it inside: models/videomae-shoplifting/pytorch_model.bin

4️⃣ Run Migrations
python manage.py migrate

5️⃣ Start the Server
python manage.py runserver
Open in browser: http://127.0.0.1:8000/

🖼️ Usage
Upload a short video clip from the browser.
Wait for the model to process it.

See result:
🚨 Alert: Shop lifter detected (Red background)
✅ Safe: Non shop lifter (Green background)

📊 Example Result

Result: Shop lifter
Confidence: 97.32% 🚨 Alert

Result: Non shop lifter
Confidence: 99.89% ✅ SAFE

👩‍💻 Developer
Eng. Tasneem Elgendy
📞 Contact: (+2) 1111126495
🌐 LinkedIn: https://www.linkedin.com/in/tasneem-elgendy-905622203
