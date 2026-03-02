
## Important Commands

pip install -r requirements.txt <br>
pip install --upgrade pip <br>
pip list <br>

&nbsp

python3 -m venv venv <br>
source venv/bin/activate <br>
uvicorn app.main:app --reload <br>
python app/download_model.py <br>

&nbsp

git remote set-url origin https://user_name@github.com/ByteForge-MP/pnb-coreai-backend.git <br>
git remote set-url origin https://token@github.com/ByteForge-MP/pnb-coreai-backend.git <br>
&nbsp
#### One step back to uncommit <br>
git reset --soft HEAD~1 <br>
&nbsp
git clone https://github.com/your-username/pnb-coreai-backend.git <br>

&nbsp

pip install transformers huggingface_hub torch accelerate <br>
pip install hf_transfer <br>
python app/download_model.py <br>
pip install --upgrade huggingface_hub <br>
huggingface_hub login <br>

&nbsp

pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


