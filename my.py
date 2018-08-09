import cv2
import os
import numpy as np

class FaceDetector:
    # Detecta a face 
    def detect_face(self, img, scaleFactor):
        # Se a imagem não é preto e branco converte para preto e branco
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: 
            gray = img
        # Importa configurações do haarcascades para não precisar treina-lo
        face_cascade = cv2.CascadeClassifier('opencv-files/haarcascades/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=4)
        # Se não encontrou faces não retorna nada
        if (len(faces) == 0):
            return None, None

        lfaces = []
        #Armazena todas as faces em e depois retorna
        for (x, y, w, h) in faces:
            # Pega somente o rosto
            roi_gray = gray[y:y+h, x:x+w]
            # Monta estrutura pra retornar as informacoes
            # Posicao 0: face - Posicao 1: local da face
            data = (roi_gray,(x,y,w,h))
            lfaces.append(data)
        return lfaces

class FaceRecognitor:
    # Inicia o reconecedor facial
    def __init__(self, configPath):
        print("[*] Carregando configuracoes ...")
        self.subjects = []
        self.detector = FaceDetector()
        self.subjects.append('')
        # Cria os reconhcecedores faciais de cada algoritmo testado
        self.face_recognizerE = cv2.face.EigenFaceRecognizer_create()
        self.face_recognizerF = cv2.face.FisherFaceRecognizer_create()
        self.face_recognizerL = cv2.face.LBPHFaceRecognizer_create()

        # Le a lista de pessoas presentes na pasta training data
        # Sendo cada um respectivo a uma pasta, por exemplo:
        # A primeira pessoa da lista vai ser ter suas imagens na pasta s1

        for line in open(configPath).readlines():
            line = line.rstrip()
            self.subjects.append(line)
                       
        print("[*] Configuracoes carregadas.")
        
        # Carrega as imagens que serão utilizadas para teste e para treinamento
        print("[*] Preparando dados ...")
        self.faces, self.labels, self.efaces = self.prepare_training_data("training-data")
        self.tfaces, self.tefaces = self.prepare_test_data("test-data")
        print("[*] Dados preparados.")

    # Desenha um retangulo na imagem
    def draw_rectangle(self, img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Desenha um texto na imagem
    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    # Desenha na imagem as faces encontradas
    def show_faces(self,img,scaleFactor):
        lfaces = self.detector.detect_face(img, scaleFactor)
        for data in lfaces:
            face, rect = data
            self.draw_rectangle(img,rect) 

    # Carrega faces das imagens de  teste 
    def prepare_test_data(self, data_folder_path):
        dirs = os.listdir(data_folder_path)

        faces = []
        efaces = []
       
        for dir_name in dirs:
            if not dir_name.startswith("s"):
                continue
            label = int(dir_name.replace("s", ""))
            subject_dir_path = data_folder_path + "/" + dir_name
            subject_images_names = os.listdir(subject_dir_path)
            
            for image_name in subject_images_names:
                if image_name.startswith("."):
                    continue
                image_path = "./"+subject_dir_path + "/" + image_name
                image = cv2.imread(image_path, 0)
                cv2.imshow('Identificando faces na imagem ...', image)
                cv2.waitKey(100)
                datas = self.detector.detect_face(image,1.02)
                for data in datas:
                    face, rect = data
                    if face is not None:
                        faces.append(face)
                        efaces.append(cv2.resize(face,(150,150)))
                    
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return faces, efaces    

    # Carrega imagens de treinamento e as labels
    def prepare_training_data(self, data_folder_path):
        dirs = os.listdir(data_folder_path)

        faces = []
        labels = []
        efaces = []
        
        # abre cada pasta e abri todas as imagens
        for dir_name in dirs:
            # toda pasta de pessoa comeca com s
            if not dir_name.startswith("s"):
                continue
            
            # pega a label da pessoa
            label = int(dir_name.replace("s", ""))
            
            # caminho para a pasta contendo as imagems
            subject_dir_path = data_folder_path + "/" + dir_name
            
            # pega o nome das imagems
            subject_images_names = os.listdir(subject_dir_path)
            sumFace = [[],[]]
            # abre cada imagem detecta a face e adiciona a uma lista
            for image_name in subject_images_names:
                
                # ignora arquivos do sistema
                if image_name.startswith("."):
                    continue
                
                # gera caminho da imagem
                image_path = "./"+subject_dir_path + "/" + image_name

                #le a imagem
                image = cv2.imread(image_path, 0)

                #mostra a imagem sendo analisada 
                cv2.imshow('Treinando na imagem ...', image)
                cv2.waitKey(100)
                
                #detecta faces
                datas = self.detector.detect_face(image,1.02)
                for data in datas:
                    face, rect = data
                    sumFace[0].append(face.shape[0])
                    sumFace[1].append(face.shape[1])
                    if face is not None:
                        faces.append(face)
                        # Guarda tambem as faces em um tamanho padrão para podermos testar
                        # os algoritmos de Eigen e o de Fisher que exigem entradas com o 
                        # mesmo tamnaho
                        efaces.append(cv2.resize(face,(150,150)))
                        labels.append(label)
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("[*] Mean faces: ", np.mean(np.asarray(sumFace[0])), np.mean(np.asarray(sumFace[1])))
        return faces, labels, efaces      

    # treina os reconhecedores com as faces utilizadas para treinamento
    def inicializeRecognizers(self):
        # Todas as funcoes abaixo recebem como entrada um lista de faces
        # e uma de labels que indica de quem é cada face
        self.face_recognizerL.train(self.faces, np.array(self.labels))
        self.face_recognizerE.train(self.efaces, np.array(self.labels))
        self.face_recognizerF.train(self.efaces, np.array(self.labels))

    # Reconhece a imagem usando o algoritmo LBPH
    def predictLBPH(self, test_img):
        #Faz uma copia para nao alterar a imagem original
        img = test_img.copy()

        #detecta as faces presentes na imagem
        datas = self.detector.detect_face(img, 1.02)
        
        # Le todas as faces presentes na imagem
        for data in datas:    
            face, rect = data

            # Reconhece a imagen na face e retorna um indice de confiabilidade
            # do reconhecedor em relação a face
            label, confidence = self.face_recognizerL.predict(face)
            
            #Pega o nome da pessoa baseado na label 
            label_text = self.subjects[label]
            
            #Desenha um retangulo ao redor da face detectada
            self.draw_rectangle(img, rect)

            #Desenha o nome da pessoa na imagem
            self.draw_text(img, label_text, rect[0], rect[1]-5)
        return img, confidence, label_text

    # Reconhece a imagem usando o algoritmo Eigen
    def predictEigen(self, test_img):
        #Faz uma copia para nao alterar a imagem original
        img = test_img.copy()

        #detecta as faces presentes na imagem
        datas = self.detector.detect_face(img, 1.02)
        
        # Le todas as faces presentes na imagem
        for data in datas:    
            face, rect = data
            # exige que as entradas tenham mesmo tamanho
            face = cv2.resize(face,(150,150))
            # Reconhece a imagen na face e retorna um indice de confiabilidade
            # do reconhecedor em relação a face
            label, confidence = self.face_recognizerE.predict(face)
            
            #Pega o nome da pessoa baseado na label 
            label_text = self.subjects[label]
            
            #Desenha um retangulo ao redor da face detectada
            self.draw_rectangle(img, rect)

            #Desenha o nome da pessoa na imagem
            self.draw_text(img, label_text, rect[0], rect[1]-5)
        return img, confidence, label_text

    # Reconhece a imagem usando o algoritmo Fisher
    def predictFisher(self, test_img):
        #Faz uma copia para nao alterar a imagem original
        img = test_img.copy()

        #detecta as faces presentes na imagem
        datas = self.detector.detect_face(img, 1.02)
        
        # Le todas as faces presentes na imagem
        for data in datas:    
            face, rect = data
            # exige que as entradas tenham mesmo tamanho
            face = cv2.resize(face,(150,150))
            # Retorn a imagen na face e retorna um indice de confiabilidade
            # do reconhecedor em relação a face
            label, confidence = self.face_recognizerF.predict(face)
            
            #Pega o nome da pessoa baseado na label 
            label_text = self.subjects[label]
            
            #Desenha um retangulo ao redor da face detectada
            self.draw_rectangle(img, rect)

            #Desenha o nome da pessoa na imagem
            self.draw_text(img, label_text, rect[0], rect[1]-5)
        return img, confidence, label_text


    # Realiza os testes e grava os arquivos em um txt
    def test(self):
        self.inicializeRecognizers()
        print("[*] Testando")

        eigenHit = 0
        ficherHit = 0
        lbphHit = 0 

        file = open('result.txt', 'w')
        file2 = open('result2.txt', 'w')
        file3 = open('result3.txt', 'w')

        file.write("Eigen\nImage   Method   Validation  Confidence Person \n")
        file2.write("Fisher\nImage   Method   Validation  Confidence Person \n")
        file3.write("LBPH\nImage   Method   Validation  Confidence Person \n")

        for i in range(1,16):
            j = i - 1
            expected = self.subjects[i]
            result1 = self.predictEigen(self.tefaces[j])
            result2 = self.predictFisher(self.tefaces[j])
            result3 = self.predictLBPH(self.tfaces[j])

            validate = []
            if result1[2] == expected:
                eigenHit += 1
                validate.append('True')
            else:
                validate.append('False')
            
            if result2[2] == expected:
                ficherHit += 1
                validate.append('True')
            else:
                validate.append('False')
            
            if result3[2] == expected:
                lbphHit += 1
                validate.append('True')
            else:
                validate.append('False')
        
            file.write(str(i)+"\tEigen\t"+validate[0]+"\t"+ str(result1[1]) + "\t" + result1[2] + "\n")
            file2.write(str(i)+"\tFisher\t"+validate[1]+"\t"+ str(result2[1]) + "\t" + result2[2] + "\n")
            file3.write(str(i)+"\tLBPH\t"+validate[2]+"\t"+ str(result3[1]) + "\t" + result3[2] + "\n")

        ehitr = eigenHit/15
        eehitr = (15 - eigenHit)/15
        ae = (eigenHit + 0)/(eigenHit + 0 + (15 - eigenHit) + 0)

        file.write('Hit rate = ' + str(ehitr))
        file.write('Error = ' + str(eehitr))
        file.write('Accuracy = ' + str(ae))
        
        fhitr = ficherHit/15
        efhitr = (15 - ficherHit)/15
        af = (ficherHit + 0)/(ficherHit + 0 + (15 - ficherHit) + 0)

        file2.write('Hit rate = ' + str(fhitr))
        file2.write('Error = ' + str(efhitr))
        file2.write('Accuracy = ' + str(af)) 

        lhitr = lbphHit/15
        elhitr = (15 - lbphHit)/15
        al = (lbphHit + 0)/(lbphHit + 0 + (15 - lbphHit) + 0)

        file3.write('Hit rate = ' + str(lhitr)+"\n")
        file3.write('Error = ' + str(elhitr)+"\n")
        file3.write('Accuracy = ' + str(al)+"\n") 

        file.close() 
        file2.close() 
        file3.close()    
        print("[*] Teste Concluido")

fr = FaceRecognitor('./config')
fr.test()