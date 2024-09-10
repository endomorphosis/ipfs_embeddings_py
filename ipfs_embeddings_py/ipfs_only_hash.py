import os
import subprocess
import tempfile
class ipfs_only_hash_py:
    def __init__(self, resources, metadata):

        return None
    
    def __call__(self, request):
        if os.path.isfile(request) == True:
            absolute_path = os.path.abspath(request)
            ipfs_hash_cmd = "bash -c 'npx ipfs-only-hash " + absolute_path 
            ipfs_hash = subprocess.check_output(ipfs_hash_cmd, shell=True).decode('utf-8').strip()
        else:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(request)
                filename = f.name
                ipfs_hash_cmd = "bash -c 'npx ipfs-only-hash " + filename + "'"
                ipfs_hash = subprocess.check_output(ipfs_hash_cmd, shell=True).decode('utf-8').strip()
        return ipfs_hash
    
    def __test__(self):
        test_file_path = "test.txt"
        test_ipfs_hash = self(test_file_path)
        print(test_ipfs_hash)
        return None
        