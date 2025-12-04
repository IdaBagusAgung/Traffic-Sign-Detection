import h5py

# Ganti 'model_rambu_lalu_lintas.h5' dengan nama file model H5 Anda
model_path = 'model_rambu_lalu_lintas.h5'

def check_h5_keys(model_path):
    try:
        with h5py.File(model_path, 'r') as model_file:
            # Mengeksplorasi kunci-kunci utama dalam file H5
            print("Main Keys:")
            print(list(model_file.keys()))

            # Mungkin perlu mengeksplorasi lebih dalam untuk menemukan kunci kelas
            # Anda bisa mengikuti kunci-kunci untuk menemukan grup/grup yang relevan
            # dan kemudian mengeksplorasi kunci-kunci dalam grup tersebut.
            
            # Contoh: Exploring a group named 'class_indices'
            # if 'class_indices' in model_file:
            #     print("Class Indices Keys:")
            #     print(list(model_file['class_indices'].keys()))

    except Exception as e:
        print(f"Error: {e}")

check_h5_keys(model_path)