import cv2, sys

# Récupération des valeurs soumises par l'utilisateur
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Lecture du classifieur de visage
face_cascade = cv2.CascadeClassifier(cascPath)

# Lecture de l'image
img = cv2.imread(imagePath)
# Convertion en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Détection des visages dans l'image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Trace un rectangle autour des visages trouvés
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


# Affiche le nombre de visages trouvés
face_text = '{0} face'.format(len(faces)) if len(
            faces) == 1 else '{0} faces'.format(len(faces))
cv2.putText(img, face_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)


# Affichage de l'image
cv2.imshow('img', img)
# On attend que l'utilisateur presse une touche
cv2.waitKey(0)
# Destruction de la fenêtre
cv2.destroyAllWindows()