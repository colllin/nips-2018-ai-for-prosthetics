import os
import uuid
import subprocess

def load_s3_model_checkpoint(policy, s3_dir, basename):
    # Create temp dir for download
    local_dir = f'./{str(uuid.uuid4())}/'
    os.makedirs(local_dir, exist_ok=True)
    
    # Download model checkpoint
    subprocess.check_output(f'aws s3 cp {s3_dir} {local_dir} --recursive --exclude "*" --include "{basename}*" --no-sign-request', shell=True)
    
    # Load model checkpoint
    policy.load(local_dir, basename)
    
    # Remove temp dir
    subprocess.check_output(f'rm -rf {local_dir}', shell=True)

def save_s3_model_checkpoint(policy, s3_dir, basename):
    # Create temp dir for download
    local_dir = f'./{str(uuid.uuid4())}/'
    os.makedirs(local_dir, exist_ok=True)

    # Save model checkpoint to disk
    policy.save(local_dir, basename)
    
    # Upload model checkpoint
    subprocess.check_output(f'aws s3 cp {local_dir}* {s3_dir} --no-sign-request', shell=True)
        
    # Remove temp dir
    subprocess.check_output(f'rm -rf {local_dir}', shell=True)
      


