from keras.models import load_model
import numpy as np
from skimage import transform
from keras.preprocessing.image import load_img

model = load_model('DLMODEL/Covid-19_and_Other_93%.h5')

diseaseLabels = {
    0: ['COVID-19', 'Pain relievers (ibuprofen or acetaminophen)  \nCough syrup or medication \nRest \nFluid intake'],
    1: ['Lung Opacity',
        'get aspiration pneumonia, you’ll be treated with strong antibiotics. You’ll also be evaluated and treated for swallowing problems, so you don’t continue to aspirate'],
    2: ['Normal', 'You are Health'],
    3: ['Pneumonia-Bacterial',
        "If you have an influenza virus,\n your doctor may prescribe medications such as oseltamivir (Tamiflu),\n zanamivir (Relenza),\n or peramivir (Rapivab).\n These drugs keep flu viruses from spreading in your body.\nIf RSV is the cause of your pneumonia,\n your doctor may prescribe a medication such as ribavirin (Virazole).\n This helps to limit the spread of viruses."],
    4: ['Pneumonia-Viral',
        'Treatment for bacterial pneumonia includes antibiotics,\n which target the specific type of bacterium causing the infection.\n A doctor might also prescribe medications to ease breathing.Additional medications may include over-the-counter (OTC) drugs to ease aches and pains,\n as well as reducing fever.Home care will often include rest and drinking plenty of fluids unless a doctor instructs otherwise.\n Be sure to finish a course of antibiotic therapy according to the doctor’s prescription,\n even if symptoms have improved.']
}


class Lung_Disease_Detector:

    def prepareImage(self, image):
        np_image = image
        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (200, 200, 1))
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

    def Classify(self, img_path) -> list:
        img = load_img(img_path)
        processed_img = self.prepareImage(img)
        prediction = model.predict(processed_img)

        for key, item in diseaseLabels.items():
            if key == np.argmax(prediction):
                return item
