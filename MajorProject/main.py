import streamlit as st
import tensorflow as tf
import numpy as np

translations = {
    'English': {
        'dashboard_title': 'Dashboard',
        'language_label': 'Choose language',
        'select_page': 'Select page',
        'home': 'Home',
        'about': 'About',
        'disease_detection': 'Disease Detection',
        'title': 'PLANT DISEASE DETECTION SYSTEM',
        'upload_image': 'Choose an Image:',
        'show_image': 'Show Image',
        'predict_disease': 'Predict Disease',
        'prediction_text': 'Our Prediction:',
        'waiting_text': 'Please wait....',
        'healthy_text': 'The {plant} plant is healthy!',
        'detected_text': 'Detected {disease} on {plant} leaf.',
        'expander_title': 'Disease details and recommendations',
        'disease_label': 'Disease:',
        'plant_label': 'Plant affected:',
        'what_it_is': 'What it is:',
        'effects': 'How it affects the plant:',
        'prevention': 'Prevention:',
        'post_treatment': 'Post-infection suggestions:',
        'home_markdown': """
    Welcome to the Plant Disease Detection System! 🌿
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Detection** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Detection** page in the sidebar to upload an image and experience the power of our Plant Disease Detection System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """,
        'about_markdown': """
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """,
    },
    'हिन्दी': {
        'dashboard_title': 'डैशबोर्ड',
        'language_label': 'भाषा चुनें',
        'select_page': 'पृष्ठ चुनें',
        'home': 'होम',
        'about': 'हमारे बारे में',
        'disease_detection': 'रोग पहचान',
        'title': 'पौधा रोग पहचान प्रणाली',
        'upload_image': 'एक छवि चुनें:',
        'show_image': 'छवि दिखाएँ',
        'predict_disease': 'रोग का पता लगाएँ',
        'prediction_text': 'हमारी पूर्वानुमान:',
        'waiting_text': 'कृपया प्रतीक्षा करें....',
        'healthy_text': '{plant} पौधा स्वस्थ है!',
        'detected_text': '{plant} पत्ती पर {disease} का पता चला।',
        'expander_title': 'रोग के विवरण और सुझाव',
        'disease_label': 'रोग:',
        'plant_label': 'प्रभावित पौधा:',
        'what_it_is': 'यह क्या है:',
        'effects': 'यह पौधे को कैसे प्रभावित करता है:',
        'prevention': 'रोकथाम:',
        'post_treatment': 'रोग-उपचार सुझाव:',
        'home_markdown': """
    पौधा रोग पहचान प्रणाली में आपका स्वागत है! 🌿
    
    हमारा लक्ष्य पौधों के रोगों की पहचान को सरल और तेज़ बनाना है। एक पौधे की छवि अपलोड करें, और हमारा सिस्टम संभावित रोगों का विश्लेषण करेगा। साथ मिलकर हम फसलों की रक्षा कर सकते हैं!

    ### यह कैसे काम करता है
    1. **छवि अपलोड करें:** **रोग पहचान** पेज पर जाएं और एक सूत्र वाली छवि अपलोड करें।
    2. **विश्लेषण:** हमारा सिस्टम रोग का पता लगाने के लिए छवि को प्रोसेस करेगा।
    3. **परिणाम:** आगे की कार्रवाई के लिए परिणाम और सुझाव देखें।

    ### क्यों चुनें?
    - **सटीकता:** हमारा सिस्टम उच्च गुणवत्ता वाली मशीन लर्निंग तकनीक का उपयोग करता है।
    - **उपयोग में आसान:** सरल और सहज इंटरफ़ेस।
    - **तेज़:** कुछ ही सेकंड में परिणाम प्राप्त करें।

    ### शुरू करें
    साइडबार में **रोग पहचान** पेज पर क्लिक करें और छवि अपलोड करके सिस्टम का उपयोग करें।
    """,
        'about_markdown': """
                #### डेटासेट के बारे में
                यह डेटासेट मूल डेटासेट से ऑफ़लाइन ऑगमेंटेशन के माध्यम से बनाया गया है। मूल डेटासेट इस github रेपो में पाया जा सकता है।
                यह डेटासेट लगभग 87K स्वस्थ और रोगग्रस्त पत्तियों की rgb छवियों से बना है, जो 38 अलग-अलग वर्गों में वर्गीकृत हैं। कुल डेटासेट को 80/20 अनुपात में प्रशिक्षण और मान्यता सेट में विभाजित किया गया है।
                भविष्यवाणी के उद्देश्य से बाद में एक नई निर्देशिका में 33 परीक्षण छवियाँ बनाई गई हैं।
                #### सामग्री
                1. प्रशिक्षण (70295 छवियाँ)
                2. परीक्षण (33 छवियाँ)
                3. मान्यता (17572 छवियाँ)
                """,
    },
}

# Tensorflow model prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model("My_new_plant_training_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)

    input_arr=np.array([input_arr]) # Convert single image to a batch
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index


def get_disease_recommendation(plant, disease):
    plant_clean = plant.replace('_', ' ').replace('(', '').replace(')', '')
    disease_clean = disease.replace('_', ' ')
    original_key = disease_clean.lower()
    alternate_key = None
    if disease_clean.lower().startswith(plant_clean.lower() + ' '):
        alternate_key = disease_clean[len(plant_clean) + 1:].lower()
        disease_clean = disease_clean[len(plant_clean) + 1:]

    disease_info = {
        'apple scab': {
            'description': 'Apple scab is a fungal leaf disease that causes dark, scabby lesions on leaves and fruit.',
            'effects': 'It reduces photosynthesis, weakens the tree, causes premature leaf drop, and lowers fruit quality.',
            'prevention': [
                'Plant resistant apple varieties and avoid overhead irrigation.',
                'Maintain good air circulation by pruning crowded branches.',
                'Remove fallen leaves and infected fruit from the orchard floor.',
            ],
            'post_treatment': [
                'Apply fungicide sprays early in the season according to label instructions.',
                'Remove and destroy infected leaves and fruit to reduce future spore loads.',
                'Keep the tree healthy with balanced fertilization and regular watering.',
            ],
        },
        'black rot': {
            'description': 'Black rot is a fungal disease that causes dark lesions on fruit, leaves, and stems.',
            'effects': 'It can cause fruit rot, leaf spots, defoliation, and general decline in plant vigor.',
            'prevention': [
                'Prune and destroy infected canes, leaves, and fruit.',
                'Avoid planting near old infected material and improve air flow.',
                'Use certified disease-free planting material.',
            ],
            'post_treatment': [
                'Apply recommended fungicides in wet weather periods.',
                'Sanitize pruning tools between cuts to prevent spread.',
                'Monitor plants frequently and remove new infections promptly.',
            ],
        },
        'cedar apple rust': {
            'description': 'Cedar apple rust is a fungal disease that spreads between apple and cedar trees.',
            'effects': 'It causes orange rust spots on leaves and can compromise fruit quality and tree health.',
            'prevention': [
                'Keep apples away from cedar and juniper hosts if possible.',
                'Remove infected leaves and twigs from apple trees.',
                'Choose resistant apple varieties when planting.',
            ],
            'post_treatment': [
                'Apply fungicides at key infection times, especially in spring.',
                'Prune out heavily infected branches on both host species.',
                'Maintain tree health through adequate watering and nutrition.',
            ],
        },
        'powdery mildew': {
            'description': 'Powdery mildew is a fungal infection that appears as a white, powdery coating on leaves.',
            'effects': 'It reduces photosynthesis, distorts new growth, and weakens the plant over time.',
            'prevention': [
                'Ensure good air circulation and avoid overcrowding plants.',
                'Water at the base of plants instead of overhead.',
                'Use resistant varieties when available.',
            ],
            'post_treatment': [
                'Apply sulfur-based or potassium bicarbonate sprays as needed.',
                'Remove severely infected leaves to reduce spread.',
                'Maintain balanced plant nutrition and avoid excess nitrogen.',
            ],
        },
        'cercospora leaf spot gray leaf spot': {
            'description': 'This fungal disease causes round spots on leaves and may reduce yield in maize.',
            'effects': 'Leaves lose green area, causing reduced photosynthesis and weaker plants.',
            'prevention': [
                'Rotate crops and remove crop debris from the field.',
                'Plant resistant hybrids or varieties if available.',
                'Avoid excessive plant density and keep fields well ventilated.',
            ],
            'post_treatment': [
                'Use fungicide treatments when conditions favor disease development.',
                'Harvest promptly when grain is mature to minimize losses.',
                'Monitor fields and remove heavily infected leaves if practical.',
            ],
        },
        'common rust': {
            'description': 'Common rust is a fungal disease on maize that produces red-brown pustules on leaves.',
            'effects': 'It decreases leaf area and can reduce grain fill in severe infections.',
            'prevention': [
                'Plant resistant maize varieties and avoid plant stress.',
                'Rotate crops to reduce fungal buildup in soil.',
                'Manage irrigation to avoid prolonged leaf wetness.',
            ],
            'post_treatment': [
                'Treat with appropriate fungicides if infection is severe.',
                'Remove crop residue and practice good field hygiene.',
                'Keep plants healthy with proper nutrition and watering.',
            ],
        },
        'northern leaf blight': {
            'description': 'Northern leaf blight causes long, grayish-brown lesions on maize leaves.',
            'effects': 'It reduces photosynthesis, weakens plants, and can lower yield.',
            'prevention': [
                'Plant resistant hybrids and rotate crops.',
                'Avoid overhead irrigation and improve airflow.',
                'Remove and destroy infected plant debris.',
            ],
            'post_treatment': [
                'Apply fungicides as recommended in areas with known disease pressure.',
                'Monitor leaf development and remove severely affected sections.',
                'Support plant health with regular fertilization and water management.',
            ],
        },
        'haunglongbing (citrus greening)': {
            'description': 'Huanglongbing is a bacterial disease that severely weakens citrus trees and ruins fruit quality.',
            'effects': 'It causes yellowing leaves, misshapen fruit, branch dieback, and eventual tree decline.',
            'prevention': [
                'Use disease-free nursery stock and control the psyllid insect vector.',
                'Remove infected trees quickly to prevent spread.',
                'Maintain tree vigor with proper watering and nutrition.',
            ],
            'post_treatment': [
                'There is no cure; focus on controlling spread and keeping healthy trees productive.',
                'Use nutritional sprays to support the tree while managing symptoms.',
                'Monitor regularly and remove severely infected trees if necessary.',
            ],
        },
        'bacterial spot': {
            'description': 'Bacterial spot is a bacterial leaf disease that causes dark, water-soaked spots on leaves and fruit.',
            'effects': 'It can defoliate plants, scar fruit, and reduce yield and marketability.',
            'prevention': [
                'Use disease-free seed and avoid overhead watering.',
                'Space plants for good air circulation and remove infected debris.',
                'Avoid working in the field when plants are wet.',
            ],
            'post_treatment': [
                'Apply copper-based sprays as recommended for bacterial diseases.',
                'Remove and destroy heavily infected leaves and fruit.',
                'Rotate crops and sanitize tools to reduce spread.',
            ],
        },
        'early blight': {
            'description': 'Early blight is a fungal disease that causes concentric rings on leaves and rots on stems and fruit.',
            'effects': 'It weakens the plant, defoliates lower leaves, and reduces yield in tomatoes and potatoes.',
            'prevention': [
                'Mulch to prevent soil splash and avoid overhead watering.',
                'Remove lower leaves and stake plants to improve air flow.',
                'Rotate crops and plant resistant varieties when possible.',
            ],
            'post_treatment': [
                'Use fungicides at early stages and remove infected plant parts.',
                'Keep foliage dry and avoid crowding plants.',
                'Maintain plant vigor with proper nutrition and irrigation.',
            ],
        },
        'late blight': {
            'description': 'Late blight is a serious fungal disease that causes dark lesions on leaves, stems, and fruit.',
            'effects': 'It can rapidly destroy potato and tomato plants under wet conditions.',
            'prevention': [
                'Use certified disease-free seed and resistant varieties.',
                'Avoid overhead irrigation and remove volunteer plants.',
                'Improve drainage and air circulation around plants.',
            ],
            'post_treatment': [
                'Apply appropriate fungicides as soon as symptoms appear.',
                'Remove and destroy infected plants to slow spread.',
                'Harvest healthy fruit promptly and avoid storing infected produce.',
            ],
        },
        'leaf mold': {
            'description': 'Leaf mold causes yellow spots and fuzzy gray growth on tomato leaves.',
            'effects': 'It reduces leaf function and weakens plants, especially in humid conditions.',
            'prevention': [
                'Grow tomatoes in full sun and provide good ventilation.',
                'Water plants at the base and avoid wetting foliage.',
                'Space plants properly and remove lower leaves.',
            ],
            'post_treatment': [
                'Remove infected leaves and use fungicide sprays if needed.',
                'Keep humidity down and avoid dense canopies.',
                'Ensure plants are healthy with regular feeding and watering.',
            ],
        },
        'septoria leaf spot': {
            'description': 'Septoria leaf spot causes small, circular spots on tomato leaves.',
            'effects': 'It defoliates plants and reduces fruit production when severe.',
            'prevention': [
                'Mulch and water at the soil level to avoid leaf splash.',
                'Rotate tomatoes and remove plant debris after harvest.',
                'Space plants for good air flow.',
            ],
            'post_treatment': [
                'Apply fungicide treatments, especially on lower leaves.',
                'Remove infected leaves and keep the plant canopy open.',
                'Maintain steady soil moisture and plant nutrition.',
            ],
        },
        'spider mites two-spotted spider mite': {
            'description': 'Spider mites are small pests that suck sap from plant leaves, causing fine webbing.',
            'effects': 'Leaves become speckled, yellow, and may fall off if infestations are heavy.',
            'prevention': [
                'Keep plants well-watered and avoid dusty conditions.',
                'Encourage beneficial insects like predatory mites.',
                'Inspect plants regularly for early signs of mites.',
            ],
            'post_treatment': [
                'Spray affected plants with water or insecticidal soap.',
                'Remove heavily infested leaves and improve humidity.',
                'Use miticides only when necessary and rotate treatment types.',
            ],
        },
        'target spot': {
            'description': 'Target spot creates dark, concentric lesions on tomato leaves and fruit.',
            'effects': 'It can defoliate plants and reduce fruit quality if left uncontrolled.',
            'prevention': [
                'Rotate crops and avoid wet foliage conditions.',
                'Apply mulch and space plants for airflow.',
                'Use disease-free seed and remove volunteer plants.',
            ],
            'post_treatment': [
                'Use fungicides when symptoms first appear.',
                'Remove and destroy infected plant material.',
                'Maintain plant health with proper watering and fertilization.',
            ],
        },
        'tomato yellow leaf curl virus': {
            'description': 'Tomato yellow leaf curl virus is a viral disease spread by whiteflies.',
            'effects': 'It causes yellowing, leaf curling, stunted growth, and severely reduced yield.',
            'prevention': [
                'Use resistant varieties and control whitefly populations.',
                'Remove infected plants immediately.',
                'Keep the growing area clean and weed-free.',
            ],
            'post_treatment': [
                'There is no cure, so focus on removing infected plants quickly.',
                'Protect healthy plants with insect netting and whitefly traps.',
                'Maintain good fertility and irrigation to support recovery of remaining plants.',
            ],
        },
        'tomato mosaic virus': {
            'description': 'Tomato mosaic virus causes mosaic patterns and distortion on tomato leaves.',
            'effects': 'It stunts plant growth and reduces fruit yield and quality.',
            'prevention': [
                'Use virus-free seed and sanitize tools frequently.',
                'Avoid handling plants when they are wet.',
                'Remove and destroy infected plants promptly.',
            ],
            'post_treatment': [
                'No cure exists; prevent spread by sanitation and removal.',
                'Disinfect tools and hands before working with healthy plants.',
                'Keep garden areas clean and minimize mechanical transmission.',
            ],
        },
        'leaf scorch': {
            'description': 'Leaf scorch causes brown, dried edges and tips on leaves, often from stress.',
            'effects': 'Affected leaves dry out, reducing healthy foliage and plant vigor.',
            'prevention': [
                'Avoid water stress and provide consistent irrigation.',
                'Protect plants from extreme heat and wind.',
                'Keep soil mulch in place to retain moisture.',
            ],
            'post_treatment': [
                'Water plants evenly and reduce environmental stress.',
                'Remove badly scorched leaves to improve appearance.',
                'Support the plant with good nutrition and regular care.',
            ],
        },
    }

    info = disease_info.get(original_key)
    if info is None and alternate_key is not None:
        info = disease_info.get(alternate_key)

    return info or {
        'description': 'This disease is not in the current recommendation database.',
        'effects': 'The plant may experience stress, reduced growth, or leaf and fruit damage.',
        'prevention': [
            'Keep plants healthy with proper watering, nutrition, and spacing.',
            'Remove infected leaves or plants quickly.',
            'Practice crop rotation and good field hygiene.',
        ],
        'post_treatment': [
            'Monitor the plant closely for changes.',
            'Apply general fungicide or bactericide products if appropriate.',
            'Improve overall plant care and reduce stress.',
        ],
    }

# Sidebar

if 'language' not in st.session_state:
    st.session_state.language = 'English'

languages = ['English', 'हिन्दी']
st.session_state.language = st.sidebar.selectbox(
    translations[st.session_state.language]['language_label'],
    languages,
    index=languages.index(st.session_state.language),
)
lang = st.session_state.language
t = translations[lang]

st.sidebar.title(t['dashboard_title'])
app_mode = st.sidebar.selectbox(t['select_page'], [t['home'], t['about'], t['disease_detection']])

#Home page
if app_mode == t['home']:
    st.header(t['title'])
    image_path=r"C:\Users\Sachin Gola\OneDrive\Desktop\Major Project\MajorProject\home_page.jpeg"
    st.image(image_path, width="stretch")
    st.markdown(t['home_markdown'])


    # About page
elif app_mode == t['about']:
    st.header(t['about'])
    st.markdown(t['about_markdown'])

elif app_mode == t['disease_detection']:
    st.header(t['disease_detection'])
    test_image = st.file_uploader(t['upload_image'])
    if st.button(t['show_image']):
        st.image(test_image, width="stretch")

    if st.button(t['predict_disease']):
        st.write(t['prediction_text'])
        with st.spinner(t['waiting_text']):
            result_index=model_prediction(test_image)
        #define class
        class_name=['Apple___Apple_scab',
         'Apple___Black_rot',
         'Apple___Cedar_apple_rust',
         'Apple___healthy',
         'Blueberry___healthy',
         'Cherry_(including_sour)___Powdery_mildew',
         'Cherry_(including_sour)___healthy',
         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
         'Corn_(maize)___Common_rust_',
         'Corn_(maize)___Northern_Leaf_Blight',
         'Corn_(maize)___healthy',
         'Grape___Black_rot',
         'Grape___Esca_(Black_Measles)',
         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
         'Grape___healthy',
         'Orange___Haunglongbing_(Citrus_greening)',
         'Peach___Bacterial_spot',
         'Peach___healthy',
         'Pepper,_bell___Bacterial_spot',
         'Pepper,_bell___healthy',
         'Potato___Early_blight',
         'Potato___Late_blight',
         'Potato___healthy',
         'Raspberry___healthy',
         'Soybean___healthy',
         'Squash___Powdery_mildew',
         'Strawberry___Leaf_scorch',
         'Strawberry___healthy',
         'Tomato___Bacterial_spot',
         'Tomato___Early_blight',
         'Tomato___Late_blight',
         'Tomato___Leaf_Mold',
         'Tomato___Septoria_leaf_spot',
         'Tomato___Spider_mites Two-spotted_spider_mite',
         'Tomato___Target_Spot',
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
         'Tomato___Tomato_mosaic_virus',
         'Tomato___healthy']
        
        # Output Layout Formatting
        classname = class_name[result_index]
        plant, disease = classname.split("___", 1)
        plant_clean = plant.replace('_', ' ').lower()
        disease_clean = disease.replace('_', ' ')
        plant_display = plant.replace('_', ' ').replace('(', '').replace(')', '')
        if disease_clean.startswith(plant_display + ' '):
            disease_clean = disease_clean[len(plant_display) + 1:]
        if disease == "healthy":
            st.success(t['healthy_text'].format(plant=plant_clean))
        else:
            st.success(t['detected_text'].format(disease=disease_clean, plant=plant_clean))
            info = get_disease_recommendation(plant, disease)
            with st.expander(t['expander_title']):
                st.markdown(f"**{t['disease_label']}** {disease_clean}")
                st.markdown(f"**{t['plant_label']}** {plant.replace('_', ' ')}")
                st.markdown(f"**{t['what_it_is']}** {info['description']}")
                st.markdown(f"**{t['effects']}** {info['effects']}")
                st.markdown(f"**{t['prevention']}**")
                for item in info['prevention']:
                    st.markdown(f"- {item}")
                st.markdown(f"**{t['post_treatment']}**")
                for item in info['post_treatment']:
                    st.markdown(f"- {item}")
