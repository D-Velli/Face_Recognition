import os
import streamlit as st
import numpy as np
from PIL import Image 
import cv2
import face_recognition

# Load offline signatures
FaceSignatures = np.load('./FaceSignatures_Db2.npy')
p = "C:\\Users\\Admin\\Desktop\\Teccart\\Session #5\\Automatisation\\IA2\\Projects\\Projet_2\\Album"
image_paths = os.listdir(".\\Album")

def main():
    st.title('Facial Recognition')
    # uploder l'image
    uploaded_img = st.file_uploader("Choisissez une Image", type=['png' ,'jpg', 'jpeg'])
    if uploaded_img is not None:  
        #affichage de l'image
        img = Image.open(uploaded_img)
        st.image (img, caption=None,width=400,  use_column_width=None, clamp=False, channels='RGB', output_format='auto')

        # Convertir l'image téléchargée
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Détection des visages dans l'image téléchargée
        face_locations = face_recognition.face_locations(img_cv)
        encodings = face_recognition.face_encodings(img_cv, face_locations)
        print(encodings)
        # Comparaison des visages 
        for face_encoding, face_location in zip(encodings, face_locations):
            #face_encoding = np.array(face_encoding, dtype=np.float64)
            signature = np.array(FaceSignatures[:, :-1], dtype=np.float64)
            matches = face_recognition.compare_faces(signature, face_encoding)
            if matches:
                st.write("Visages détectés:")
                for match_index, match in enumerate(matches):
                    if match:  
                        name = FaceSignatures[match_index, -1]
                        # st.write(name)
                        image_path = os.path.join(".\\Album", f"{name}.jpg")
                        family_img = Image.open(image_path)
                        print(family_img)
                        st.image(family_img, caption=None, width=200, clamp=False, channels='RGB')
            else:
                st.write("Visage inconnu")
                print("y'a nada")
    
if __name__ == '__main__':
    main()
