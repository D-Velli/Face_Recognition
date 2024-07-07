import cv2, numpy as np, face_recognition, os;
# Image datase path 
path='.\\Album'
#print(path)
# Global variables
image_list=[]# list of images
name_list=[]#list of image names
# Gral all images from the folder
myList= os.listdir(path)

#Load images
for img in myList:
    curImg = cv2.imread(os.path.join(path, img))
    image_list.append(curImg)
    imgName= os.path.splitext(img)[0]
    name_list.append(imgName)

# Define a function to detect face and extract features thereform
def findEncoding(img_list, imgName_list):
    '''
    summary 
     Define a function to detect face and extract features thereform
     Args:
         img_list (list): list of BGR images
         ImgName_list (list) :  Liste of image name
    '''  
    signatures_DB = []
    count=1
    for myimg, name in zip(img_list, imgName_list):
        img=cv2.cvtColor(myimg, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        for encoding in face_encodings:
            signature_class = encoding.tolist() + [name]
            signatures_DB.append(signature_class)
        # if len(face_encodings) > 0:
        #     signature = face_encodings[0]  
        #     signature_class = signature.tolist() + [name]
        #     signatures_DB.append(signature_class)
        # else:
        #     print(f"Aucun visage n'a été détecté dans l'image {name}")
        # # signature_class = signature.tolist() + [name]
        # # signatures_DB.append(signature_class)
        print(f'{int((count /(len(img_list))))*100} % extracted')
        count+=1
    signatures_DB= np.array(signatures_DB) 
    np.save('FaceSignatures_DB2.npy', signatures_DB)  
    print('Signature_db stored')

def main():
    findEncoding(image_list,name_list)

if __name__=='__main__':
    main()    