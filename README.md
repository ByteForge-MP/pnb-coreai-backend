
pip install -r requirements.txt
pip install --upgrade pip
pip list

python3 -m venv venv
source venv/bin/activate
uvicorn app.main:app --reload

git remote set-url origin https://user_name@github.com/ByteForge-MP/pnb-coreai-backend.git

git remote set-url origin https://token@github.com/ByteForge-MP/pnb-coreai-backend.git


---------Model-------------

pip install transformers huggingface_hub torch accelerate
pip install hf_transfer
python app/download_model.py

pip install --upgrade huggingface_hub
huggingface_hub login/ or run code file
python app/download_model.py




--------Git--------------------
killall git

