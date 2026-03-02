
pip install -r requirements.txt <br>
pip install --upgrade pip <br>
pip list <br>

python3 -m venv venv <br>
source venv/bin/activate <br>
uvicorn app.main:app --reload <br>

git remote set-url origin https://user_name@github.com/ByteForge-MP/pnb-coreai-backend.git <br>
git remote set-url origin https://token@github.com/ByteForge-MP/pnb-coreai-backend.git <br>


---------Model-------------

pip install transformers huggingface_hub torch accelerate
pip install hf_transfer
python app/download_model.py

pip install --upgrade huggingface_hub
huggingface_hub login/ or run code file
python app/download_model.py




--------Git--------------------
killall git
rm .git/index.lock   

brew install git-lfs
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git add .gitattributes
git add app/models/smollm2-1.7b-local/
git commit -m "Add SmolLM2 local weights via LFS"
git push origin main

# One step back to uncommit
git reset --soft HEAD~1

For fetching:
git clone https://github.com/your-username/pnb-coreai-backend.git
cd pnb-coreai-backend
git lfs pull


-----------------

pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

-----------------
