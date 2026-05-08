import streamlit as st
import tensorflow as tf
import numpy as np
import os
import base64

# --- 1. UI TEXT TRANSLATIONS ---
st.set_page_config(page_title="Plant Disease Detection And Recovery System", layout="wide") # Hardcoded title for now
if 'active_tab_key' not in st.session_state:
    st.session_state.active_tab_key = "home_page_option"
# --- CSS STYLING ---
st.markdown("""
    <style>
    /* 1. Force all text to be Gold and White with shadows */
    h1, h2, h3 {
        color: #F4D03F !important;
        text-shadow: 2px 2px 4px #000000 !important;
    }
    p, li, span, label, .stMarkdown {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px #000000 !important;
    }

    /* 2. Style the File Uploader Box */
    [data-testid="stFileUploader"] {
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 2px dashed #F4D03F !important;
        border-radius: 15px;
        padding: 20px;
    }

    /* 3. Style the "Browse files" button inside the uploader */
    [data-testid="stFileUploader"] button {
        background-color: #F4D03F !important;
        color: black !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
    }

    /* 4. Style the navigation buttons at the top */
    .stButton>button {
        color: #F4D03F !important;
        background-color: rgba(0, 0, 0, 0.6) !important;
        border: 2px solid #F4D03F !important;
        font-weight: bold;
        border-radius: 10px;
    }

    /* 5. Fix the Alert box (Please upload image) */
    .stAlert {
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: #F4D03F !important;
        border: 1px solid #F4D03F !important;
    }
     /* 6. Side-by-side Columns Styling */
    [data-testid="column"] {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- EMBEDDED TRANSLATIONS DICTIONARY ---
TRANSLATIONS = {
    "en": {
        "app_title": "Plant Disease Detection And Recovery System", # Can be used later if you want dynamic tab title
        "home_header": "PLANT DISEASE DETECTION AND RECOVERY SYSTEM",
        "home_welcome_greeting": "Welcome to the PLANT DISEASE DETECTION AND RECOVERY SYSTEM! 🌱🔍",
        "home_mission_statement": "My mission is to empower farmers, gardeners, and agricultural enthusiasts with an accessible tool for the preliminary identification of plant diseases and recovery methods.",
        "home_how_it_works_header": "How It Works",
        "home_step1_upload": "1. **Navigate & Upload:** Go to the 'Disease Recognition' page and upload a clear image of a plant leaf showing potential symptoms.",
        "home_step2_analyze": "2. **Smart Analysis:** Our DL model, trained on thousands of images, will analyze your photo to identify patterns associated with various plant conditions.",
        "home_step3_results": "3. **Get Insights:** Receive a prediction for the potential disease, along with detailed symptoms, preventative measures (both cultural and chemical), and general recommendations.",
        "home_multilingual_note":" 4. **This application:** supports English, Amharic, and Tigrinya. You can switch languages using the button at the top.",
        #"home_disclaimer": "**Disclaimer:** This tool provides preliminary guidance and is not a substitute for professional agronomic advice. For critical concerns, please consult with a local agricultural expert.",
        "home_call_to_action": "Let's protect our crops and ensure healthier harvests together! Get started by navigating to the 'Disease Recognition' page.",

        "about_header": "About This Project",
        "about_introduction_header": "Introduction",
        "about_introduction_text": "This PLANT DISEASE DETECTION AND RECOVERY SYSTEM was developed to provide an accessible and informative tool for identifying common plant diseases. By leveraging advancements in deep learning and web technologies, we aim to support agricultural communities and gardening enthusiasts in maintaining plant health.",
        "about_technology_header": "Technology Used",
        "about_tech_deep_learning": "- **Deep Learning Model:** A Convolutional Neural Network (CNN) was designed and trained from scratch using Python, TensorFlow, and Keras. It's capable of classifying 38 different plant conditions.",
        "about_tech_webapp": "- **Web Application:** The user interface is built with Streamlit, a Python framework that allows for rapid development of interactive data applications.",
        "about_tech_language": "- **Multilingual Support:** The application interface and informational content are available in English, Amharic, and Tigrinya, implemented using a custom translation management system.",
        "about_dataset_header": "About the Dataset",
        "about_dataset_description": "The model was trained on a comprehensive dataset of approximately 87,000 RGB images of plant leaves. This dataset covers 38 distinct classes, including various diseases and healthy plant leaves across multiple crop types. The data was carefully preprocessed (resized, normalized) and split into training and validation sets to ensure robust model learning and evaluation. Augmentation techniques were also explored to improve generalization.",
        "about_features_header": "Key Features",
        "about_feature_prediction": "- **Disease Prediction:** Upload a leaf image to get an AI-powered disease prediction.",
        "about_feature_info": "- **Detailed Information:** Access information on symptoms, cultural and chemical prevention methods, and recommendations for identified diseases.",
        "about_feature_multilingual": "- **Multilingual Access:** Full application support in English, Amharic, and Tigrinya.",
        "about_developer_header": "About the Developer",
        "about_developer_text": "This project by **Tedros Nigus** from Grade 10 Boarding School  is an independent project to explore AI in agriculture. My goal is to develop an automated system that can accurately identify common plant diseases from leaf images using a deep learning model and provide functional information in multiple languages.",
        "about_future_scope_header": "Future Scope",
        "about_future_scope_text": "Potential future enhancements include expanding the disease database, incorporating disease severity assessment, and exploring offline mobile capabilities.",

        "dashboard_title": "Dashboard", # For sidebar title
        "select_page_label": "Select Page", # For page selection dropdown label
        "home_page_option": "Home", # Option text for Home
        "about_page_option": "About", # Option text for About
        "disease_recognition_page_option": "Disease Recognition", # Option text for DR
        "file_uploader_main_label": "Upload a leaf image for analysis:",
        "file_uploader_help_text": "Accepted formats: JPG, JPEG, PNG. Max file size: 200MB.",
        "recognition_header": "Disease Recognition",
        "upload_prompt": "Choose an Image of a plant leaf:",
        "show_image_button": "Show Image",
        "predict_button": "Predict Disease",
        "model_predict_msg": "Model is predicting it's a: {disease_name}", # {} is placeholder for disease name
        "spinner_text": "Please wait...",
        "recommendations_subheader": "Recommendations & Information",
        "description_label": "Description",
        "symptoms_label": "Symptoms",
        "further_info_label": "Further Information",
        "expert_consultation_disclaimer": "Note: This information is for general guidance. Always consult with a local agricultural expert for specific advice and treatment options suitable for your region and conditions.",
        "no_recommendation_available": "Detailed information for this disease or in your selected language is not yet available.",
        "language_selectbox_label":  "Language", # Label for the language dropdown (if you add one)
        "language_switch_button_text": "Switch to {next_lang_name}", # For the language switch button
        "lang_name_en": "English",
        "lang_name_am": "Amharic",
        "lang_name_ti": "Tigrigna",
        "error_model_load": "Critical error: Model could not be loaded. Application functionality will be limited.",
        "error_model_file_not_found": "Model file not found at '{model_path}'. Please ensure it's in the correct location.",
        "error_prediction_failed": "Prediction failed. Please check logs or try another image.",
        "info_upload_image": "Please upload an image for prediction.",
        "error_prediction_index_range": "Prediction index out of range. Please check model output and class_name list."
    },
    "am": { 
        "app_title": "የእፅዋት በሽታን ለይቶ ማወቅ እና ማገገሚያ ስርዓት",
        "home_header": "የእፅዋት በሽታን ለይቶ ማወቅ እና ማገገሚያ ስርዓት",
        "home_welcome_greeting": "እንኳን ወደ የእፅዋት በሽታን ለይቶ ማወቅ እና ማገገሚያ ስርዓት በደህና መጡ! 🌱🔍",
        "home_mission_statement": "የእኛ ተልዕኮ ገበሬዎችን፣ አትክልተኞችን እና የግብርና አፍቃሪዎችን የተክል በሽታዎችን የመጀመሪያ ደረጃ ለመለየት በሚያስችል ተደራሽ መሳሪያ ማብቃት ነው።",
        "home_how_it_works_header": "እንዴት እንደሚሰራ",
        "home_step1_upload": "1. **ወደ ገጽ ይሂዱ እና ይስቀሉ፦** ወደ 'የበሽታ መለያ' ገጽ ይሂዱና ሊሆኑ የሚችሉ የበሽታ ምልክቶችን የሚያሳይ የተክል ቅጠል ንጹሕ ምስል ይስቀሉ።",
        "home_step2_analyze": "2. **ብልህ ትንተና፦** በሺዎች በሚቆጠሩ ምስሎች የሰለጠነው የእኛ የሰው ሰራሽ የማሰብ ችሎታ (AI) ሞዴል፣ ከተለያዩ የተክል ሁኔታዎች ጋር የተያያዙ ቅርጾችን ለመለየት ፎቶዎን ይመረምራል።",
        "home_step3_results": "3. **ግንዛቤዎችን ያግኙ፦** ሊኖር ስለሚችል በሽታ ትንበያ፣ ዝርዝር ምልክቶች፣ የመከላከያ እርምጃዎች (ባህላዊ እና ኬሚካዊ) እና አጠቃላይ ምክሮችን ይቀበሉ።",
        "home_multilingual_note": "4. **ይህ መተግበሪያ፦** እንግሊዝኛ፣ አማርኛ እና ትግርኛን ይደግፋል። ከላይ ያለውን ቁልፍ በመጠቀም ቋንቋዎችን መቀየር ይችላሉ።",
        #"home_disclaimer": "**ማስተባበያ፦** ይህ መሳሪያ የመጀመሪያ ደረጃ መመሪያን ይሰጣል እና የባለሙያ የግብርና ምክርን አይተካም። ለከባድ ስጋቶች፣ እባክዎ የአካባቢዎን የግብርና ባለሙያ ያማክሩ።",
        "home_call_to_action": "ሰብሎቻችንን እንጠብቅ እና ጤናማ ምርትን በጋራ እናረጋግጥ! ወደ 'የበሽታ መለያ' ገጽ በመሄድ ይጀምሩ።",
        "about_header": "ስለዚህ ፕሮጀክት",
        "about_introduction_header": "መግቢያ",
        "about_introduction_text": "ይህ የእፅዋት በሽታን ለይቶ ማወቅ እና ማገገሚያ ስርዓት የተለመዱ የተክል በሽታዎችን ለመለየት ተደራሽ እና መረጃ ሰጪ መሳሪያ ለማቅረብ ተዘጋጅቷል። በሰው ሰራሽ የማሰብ (Deep Learning) እና በድር ቴክኖሎጂዎች ውስጥ ያሉትን እድገቶች በመጠቀም፣ የግብርና ማህበረሰቦችን እና የአትክልት አፍቃሪዎችን የተክል ጤናን ለመጠበቅ እንደግፋለን።",
        "about_technology_header": "ጥቅም ላይ የዋለ ቴክኖሎጂ",
        "about_tech_deep_learning": "- **የሰው ሰራሽ የማሰብ (Deep Learning) ሞዴል፦** የኮንቮሉሽናል ኒውራል ኔትዎርክ (CNN) ከባዶ በፓይዘን፣ ቴንሰርፍሎው እና ኬራስ በመጠቀም ተቀርጾ የሰለጠነ ነው። 38 የተለያዩ የተክል ሁኔታዎችን መመደብ ይችላል።",
        "about_tech_webapp": "- **የድር መተግበሪያ፦** የተጠቃሚ በይነገጽ በስትሪምሊት የተገነባ ሲሆን፣ ይህም መስተጋብራዊ የመረጃ መተግበሪያዎችን በፍጥነት ለማዳበር የሚያስችል የፓይዘን ማዕቀፍ ነው።",
        "about_tech_language": "- **የብዙ ቋንቋ ድጋፍ፦** የመተግበሪያው በይነገጽ እና የመረጃ ይዘት በእንግሊዝኛ፣ በአማርኛ እና በትግርኛ የሚገኝ ሲሆን፣ ብጁ የትርጉም አስተዳደር ስርዓትን በመጠቀም ተተግብሯል።",
        "about_dataset_header": "ስለ ዳታሴቱ",
        "about_dataset_description": "ሞዴሉ ወደ 87,000 በሚጠጉ የ RGB የተክል ቅጠል ምስሎች አጠቃላይ ዳታሴት ላይ የሰለጠነ ነው። ይህ ዳታሴት 38 የተለያዩ ክፍሎችን ያካተተ ሲሆን፣ እነዚህም የተለያዩ በሽታዎችን እና ጤናማ የተክል ቅጠሎችን በተለያዩ የሰብል ዓይነቶች ላይ ያጠቃልላል። መረጃው በጥንቃቄ ቅድመ-ሂደት ተደርጎበታል (መጠን ተቀይሯል፣ መደበኛ ሆኗል) እና ጠንካራ የሞዴል ትምህርት እና ግምገማን ለማረጋገጥ ወደ ስልጠና እና ማረጋገጫ ስብስቦች ተከፍሏል። አጠቃላይነትን ለማሻሻል የማበልጸጊያ ዘዴዎችም ተዳሰዋል።",
        "about_features_header": "ቁልፍ ባህሪያት",
        "about_feature_prediction": "- **የበሽታ ትንበያ፦** በሰው ሰራሽ የማሰብ (AI) የተደገፈ የበሽታ ትንበያ ለማግኘት የቅጠል ምስል ይስቀሉ።",
        "about_feature_info": "- **ዝርዝር መረጃ፦** በታወቁ በሽታዎች ላይ ስላሉ ምልክቶች፣ ባህላዊ እና ኬሚካዊ የመከላከያ ዘዴዎች እና ምክሮች መረጃ ያግኙ።",
        "about_feature_multilingual": "- **የብዙ ቋንቋ ተደራሽነት፦** ሙሉ የመተግበሪያ ድጋፍ በእንግሊዝኛ፣ በአማርኛ እና በትግርኛ።",
        "about_developer_header": "ስለ አዘጋጁ",
        "about_developer_text": "ይህ ፕሮጀክት በ **ቴድሮስ ንጉስ** ከ10ኛ ክፍል አዳሪ ትምህርት ቤት በግብርና ውስጥ AIን ለመዳሰስ ራሱን ችሎ የተዘጋጀ ፕሮጀክት ነው። አላማዬ ጥልቅ የመማሪያ ሞዴልን በመጠቀም የተለመዱ የዕፅዋት በሽታዎችን ከቅጠል ምስሎች በትክክል ለመለየት እና ተግባራዊ መረጃዎችን በበርካታ ቋንቋዎች ለማቅረብ የሚያስችል አውቶሜትድ ስርዓት ማዘጋጀት ነው።",
        "about_future_scope_header": "የወደፊት ዕቅድ",
        "about_future_scope_text": "ሊሆኑ የሚችሉ የወደፊት ማሻሻያዎች የበሽታ ዳታቤዝ ማስፋፋት፣ የበሽታን የክብደት መጠን መገምገም እና ከመስመር ውጭ የሞባይል ችሎታዎችን መዳሰስን ያካትታሉ።",
        "dashboard_title": "ዳሽቦርድ",
        "select_page_label": "ገጽ ይምረጡ",
        "home_page_option": "ዋና ገጽ",
        "about_page_option": "ስለ",
        "disease_recognition_page_option": "የበሽታ መለያ",
        "recognition_header": "የበሽታ መለያ",
        "file_uploader_main_label": "ለመተንተን የቅጠል ምስል ይስቀሉ፡",
        "file_uploader_help_text": "የሚፈቀዱ ቅርጸቶች፦ JPG፣ JPEG፣ PNG። ከፍተኛ የፋይል መጠን፦ 200MB።",
        "upload_prompt": "የተክል ቅጠል ምስል ይምረጡ:",
        "show_image_button": "ምስል አሳይ",
        "predict_button": "በሽታውን ተንብይ",
        "model_predict_msg": "ሞዴሉ እየተነበየ ነው፡ {disease_name}",
        "recommendations_subheader": "ምክሮች እና መረጃ",
        "description_label": "መግለጫ",
        "symptoms_label": "ምልክቶች",
        "further_info_label": "ተጨማሪ መረጃ",
        "expert_consultation_disclaimer": "ማሳሰቢያ፦ ይህ መረጃ ለአጠቃላይ መመሪያ ነው። ለእርስዎ ክልል እና ሁኔታ ተስማሚ ለሆኑ የተወሰኑ ምክሮች እና የሕክምና አማራጮች ሁልጊዜ የአካባቢዎን የግብርና ባለሙያ ያማክሩ።",
        "no_recommendation_available": "ለዚህ በሽታ ወይም በመረጡት ቋንቋ ዝርዝር መረጃ በአሁኑ ጊዜ አይገኝም።",
        "spinner_text": "እባክዎ ይጠብቁ...",
        "language_selectbox_label": "ቋንቋ",
        "language_switch_button_text": "ወደ {next_lang_name} ቀይር",
        "lang_name_en": "እንግሊዝኛ",
        "lang_name_am": "አማርኛ",
        "lang_name_ti": "ትግርኛ",
        "error_model_load": "ከባድ ስህተት፦ ሞዴሉ ሊጫን አልቻለም። የመተግበሪያው ተግባራዊነት የተገደበ ይሆናል።",
        "error_model_file_not_found": "የሞዴል ፋይል በ '{model_path}' አልተገኘም። እባክዎ በትክክለኛው ቦታ መሆኑን ያረጋግጡ።",
        "error_prediction_failed": "ትንበያ አልተሳካም። እባክዎ ሎጎችን ይመልከቱ ወይም ሌላ ምስል ይሞክሩ።",
        "info_upload_image": "እባክዎ ለትንበያ ምስል ይስቀሉ።",
        "error_prediction_index_range": "የትንበያ ኢንዴክስ ከክልል ውጪ ነው። እባክዎ የሞዴሉን ውጤት እና የክፍል ስም ዝርዝር ይመልከቱ።"
    },
    "ti": {
        "app_title": "ስርዓት ምፍላጥን ምሕዋይን ሕማማት ተኽሊ",
        "home_header": "ስርዓት ምፍላጥን ምሕዋይን ሕማማት ተኽሊ",
        "home_welcome_greeting": "እንቋዕ ናብ ስርዓት ምፍላጥን ምሕዋይን ሕማማት ተኽሊ ብደሓን መጻእኩም! 🌱🔍",
        "home_mission_statement": "ተልእኾና ንሓረስቶት፣ ኣትክልተኛታትን ኣፍቀርቲ ሕርሻን ብቐሊሉ ተበጻሒ ዝኾነ ናይ ተኽሊ ሕማማት መጀመርታ መለለዪ መሳርሒ ምሃብ እዩ።",
        "home_how_it_works_header": "ከመይ ይሰርሕ",
        "home_step1_upload": "1. **ናብ ገጽ ብምኻድ ስቐሉ፦** ናብ 'ሕማም ምልላይ' ገጽ ብምኻድ፡ ናይ ዝኾነ ይኹን ናይ ሕማም ምልክታት ዘርኢ ናይ ተኽሊ ቆጽሊ ንጹር ስእሊ ስቐሉ።",
        "home_step2_analyze": "2. **ብልሓት ዘለዎ ትንተና፦** ናይና ብኣሽሓት ዝቑጸሩ ስእልታት ዝሰልጠነ ናይ ሰብ-ሰርሖ ብልሒ (AI) ሞዴል፡ ምስ ዝተፈላለዩ ናይ ተኽሊ ኩነታት ዝተኣሳሰሩ ቅርጽታት ንምልላይ ነቲ ስእልኹም ክትንትኖ እዩ።",
        "home_step3_results": "3. **ግንዛበታት ኣክቡ፦** ብዛዕባ ዝኽእል ሕማም ትንቢት፡ ዝርዝራዊ ምልክታት፡ መከላኸሊ ስጉምትታት (ባህላውን ኬሚካላውን) ከምኡ'ውን ሓፈሻዊ ምኽርታት ተቐበሉ።",
        "home_multilingual_note": " 4. **እዚ መተግበሪ፦**  እንግሊዝኛ፣ ኣምሓርኛን ትግርኛን ይድግፍ። ኣብ ላዕሊ ዘሎ መላግቦ ብምጥቃም ቋንቋታት ክትቅይሩ ትኽእሉ ኢኹም።",
        #"home_disclaimer": "**ማሕበር፦** እዚ መሳርሒ መጀመርታ መምርሒ ይህብ እምበር ናይ ሞያውያን ናይ ሕርሻ ምኽሪ ኣይትክእን። ንኸቢድ ስክፍታታት፡ በጃኹም ምስ ናይ ከባቢኹም ናይ ሕርሻ ሞያዊ ተመኸሩ።",
        "home_call_to_action": "ሰብልና ንከላኸልን ጥዕና ዘለዎ ምህርቲ ብሓባር ነረጋግጽን! ናብ 'ሕማም ምልላይ' ገጽ ብምኻድ ጀምሩ።",
        "about_header": "ብዛዕባ እዚ ፕሮጀክት",
        "about_introduction_header": "መእተዊ",
        "about_introduction_text": "እዚ ስርዓት ምፍላጥን ምሕዋይን ሕማማት ተኽሊ፡ ንልሙዳት ናይ ተኽሊ ሕማማት ንምልላይ ተበጻሕን መረዳእታ ዝህብን መሳርሒ ንምቕራብ ዝተሰርሐ እዩ። ምዕባለታት ናይ ሰብ-ሰርሖ ብልሒ (Deep Learning) ከምኡ'ውን ናይ ዌብ ቴክኖሎጂታት ብምጥቃም፡ ንማሕበረሰባት ሕርሻን ኣፍቀርቲ ኣታኽልትን ኣብ ምሕላው ጥዕና ተኽሊ ንምድጋፍ ንጽዕር።",
        "about_technology_header": "ዝተጠቐምናሉ ቴክኖሎጂ",
        "about_tech_deep_learning": "- **ናይ ሰብ-ሰርሖ ብልሒ (Deep Learning) ሞዴል፦** ኮንቮሉሽናል ኒውራል ኔትዎርክ (CNN) ካብ ባዶ ብፓይዘን፡ ቴንሰርፍሎውን ኬራስን ተሰሪሑን ሰልጢኑን። 38 ዝተፈላለዩ ናይ ተኽሊ ኩነታት ክምድብ ይኽእል።",
        "about_tech_webapp": "- **ናይ ዌብ መተግበሪ፦** እቲ ናይ ተጠቀምቲ መተሓባበሪ ብስትሪምሊት ዝተሰርሐ ኮይኑ፡ እዚ ድማ መስተጋብራዊ ናይ ዳታ መተግበሪታት ብቕልጡፍ ንምስራሕ ዘኽእל ናይ ፓይዘን ማዕቀፍ እዩ።",
        "about_tech_language": "- **ናይ ብዙሓት ቋንቋታት ድጋፍ፦** እቲ ናይ መተግበሪ መተሓባበርን ናይ መረዳእታ ትሕዝቶን ብእንግሊዝኛ፡ ኣምሓርኛን ትግርኛን ይርከብ፡ እዚ ድማ ብልምዲ ናይ ትርጉም ኣመራርሓ ስርዓት ተተግቢሩ።",
        "about_dataset_header": "ብዛዕባ እቲ ዳታሴት",
        "about_dataset_description": "እቲ ሞዴል ኣስታት 87,000 ዝኾኑ ናይ RGB ናይ ተኽሊ ቆጽሊ ስእልታት ዘለዎ ሰፊሕ ዳታሴት ተጠቒሙ ሰልጢኑ። እዚ ዳታሴት 38 ዝተፈላለዩ ክፍለታት ዝሽፍን ኮይኑ፡ እዚኦም ድማ ዝተፈላለዩ ሕማማትን ጥዑያት ናይ ተኽሊ ቆጽልን ኣብ ዝተፈላለዩ ዓይነታት ሰብሊ የጠቓልሉ። እቲ ዳታ ብጥንቃቐ ቅድመ-መስርሕ ተገይሩሉ (ስፍሓቱ ተስተኻኺሉ፡ መደበኛ ኮይኑ) ከምኡ'ውን ርግጸኛ ናይ ሞዴል ምምሃርን ገምጋምን ንምርግጋጽ ናብ ናይ ስልጠናን መረጋገጺን ስብስባት ተመቓቒሉ። ኣጠቃላይነት ንምምሕያሽ ናይ ምዕባይ ቴክኒካት እውን ተፈቲኖም።",
        "about_features_header": " ቀንዲ ባህርያት",
        "about_feature_prediction": "- **ናይ ሕማም ትንቢት፦** ብናይ ሰብ-ሰርሖ ብልሒ (AI) ዝተደገፈ ናይ ሕማም ትንቢት ንምርካብ ናይ ቆጽሊ ስእሊ ስቐሉ።",
        "about_feature_info": "- **ዝርዝራዊ መረዳእታ፦** ብዛዕባ ዝተለለዩ ሕማማት፡ ምልክታቶም፡ ባህላውን ኬሚካላውን መከላኸሊ ሜላታትን ምኽርታትን መረዳእታ ኣክቡ።",
        "about_feature_multilingual": "- **ናይ ብዙሓት ቋንቋታት ተበጻሕነት፦** ምሉእ ናይ መተግበሪ ድጋፍ ብእንግሊዝኛ፡ ኣምሓርኛን ትግርኛን።",
        "about_developer_header": "ብዛዕባ እቲ ኣዳላዊ",
        "about_developer_text": "እዚ ፕሮጀክት ብ **ቴድሮስ ንጉስ** ካብ 10ይ ክፍሊ ኣሕዳሪ ቤት ትምህርቲ ኣብ ሕርሻ AI ንምድህሳስ ዝዓለመ ናጻ ፕሮጀክት እዩ። ዕላማይ ዓሚቕ ትምህርቲ ሞዴል ተጠቒምካ ካብ ምስሊ ቆፅሊ ልሙዳት ሕማማት ተኽሊ ብትኽክል ከለሊ ዝኽእል ኣውቶማቲክ ስርዓት ምምዕባልን ብብዙሕ ቋንቋታት ተግባራዊ ሓበሬታ ምሃብን እዩ።",
        "about_future_scope_header": "ናይ መጻኢ ዕላማ",
        "about_future_scope_text": "ዝኽእሉ ናይ መጻኢ መመሓየሺታት፡ ምግፋሕ ናይ ሕማም ዳታቤዝ፡ ምግምጋም ናይ ሕማም ክብደትን ምድህሳስ ናይ ከመይ ጌርካ ብዘይ መስመር ሞባይል ዓቕምታትን የጠቓልሉ።",
        "dashboard_title": "ዳሽቦርድ",
        "select_page_label": "ገጽ ምረጽ",
        "home_page_option": "ቀዳማይ ገጽ",
        "about_page_option": "ብዛዕባ",
        "disease_recognition_page_option": "ሕማም ምልላይ",
        "file_uploader_main_label": "ንትንተና ናይ ቆጽሊ ስእሊ ስቐሉ:",
        "file_uploader_help_text": "ዝፍቀዱ ቅርጽታት፦ JPG፣ JPEG፣ PNG። ዝለዓለ ዓቐን ፋይል፦ 200MB።",
        "recognition_header": "ሕማም ምልላይ",
        "upload_prompt": "ናይ ተኽሊ ቆጽሊ ስእሊ ምረጽ:",
        "show_image_button": "ስእሊ ኣርኢ",
        "predict_button": "ሕማም ተነብይ",
        "model_predict_msg": "ሞዴል ዝግምት ዘሎ፡ {disease_name}",
        "recommendations_subheader": "ምኽርታትን መረዳእታን",
        "description_label": "መግለጺ",
        "symptoms_label": "ምልክታት",
        "further_info_label": "ተወሳኺ መረዳእታ",
        "expert_consultation_disclaimer": "ማሕበር፦ እዚ መረዳእታ ንሓፈሻዊ መምርሒ እዩ። ንኣከባቢኹምን ኩነታትኩምን ዝምልከት ፍሉይ ምኽርን ኣማራጺታት ሕክምናን ንምርካብ ኩሉ ግዜ ምስ ናይ ከባቢኹም ናይ ሕርሻ ክኢላ ተመኸሩ።",
        "no_recommendation_available": "ንዚ ሕማም ወይ ኣብ ዝመረጽኩምዎ ቋንቋ ዝርዝራዊ መረዳእታ ሕጂ የለን።",
        "spinner_text": "በጃኹም ተጸበዩ...",
        "language_selectbox_label": "ቋንቋ",
        "language_switch_button_text": "ናብ {next_lang_name} ቀይር",
        "lang_name_en": "እንግሊዝኛ",
        "lang_name_am": "ኣምሓርኛ",
        "lang_name_ti": "ትግርኛ",
        "error_model_load": "ከቢድጌጋ፦ ሞዴል ክጽዕን ኣይከኣለን። ናይቲ መተግበሪ ተግባራዊነት ውሱን ክኸውን እዩ።",
        "error_model_file_not_found": "ሞዴል ፋይል ኣብ '{model_path}' ኣይተረኽበን። በጃኹም ኣብቲ ትኽክለኛ ቦታ ምህላዉ ኣረጋግጹ።",
        "error_prediction_failed": "ትንቢት ኣይተሳኸዐን። በጃኹም ነቶም ሎግታት መርምሩ ወይ ካልእ ስእሊ ፈትኑ።",
        "info_upload_image": "በጃኹም ንትንቢት ዝኸውን ስእሊ ስቀሉ።",
        "error_prediction_index_range": "ናይ ትንቢት ኢንዴክስ ካብቲ ወሰን ወጻኢ እዩ። በጃኹም ነቲ ውጽኢት ሞዴልን ዝርዝር ስም ክፍልን መርምሩ።"
    }
}

DISEASE_NAME_TRANSLATIONS = {
    "en": {
        "Apple___Apple_scab": "Apple Scab", "Apple___Black_rot": "Apple Black Rot", "Apple___Cedar_apple_rust": "Apple Cedar Apple Rust", "Apple___healthy": "Healthy Apple",
        "Blueberry___healthy": "Healthy Blueberry", "Cherry_(including_sour)___Powdery_mildew": "Cherry Powdery Mildew", "Cherry_(including_sour)___healthy": "Healthy Cherry",
        "Corn_(maize)___Cerc_ospora_leaf_spot Gray_leaf_spot": "Corn Cercospora Leaf Spot/Gray Leaf Spot", "Corn_(maize)___Common_rust_": "Corn Common Rust",
        "Corn_(maize)___Northern_Leaf_Blight": "Corn Northern Leaf Blight", "Corn_(maize)___healthy": "Healthy Corn", "Grape___Black_rot": "Grape Black Rot",
        "Grape___Esca_(Black_Measles)": "Grape Esca (Black Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape Leaf Blight", "Grape___healthy": "Healthy Grape",
        "Orange___Haunglongbing_(Citrus_greening)": "Orange Huanglongbing (Citrus Greening)", "Peach___Bacterial_spot": "Peach Bacterial Spot", "Peach___healthy": "Healthy Peach",
        "Pepper,_bell___Bacterial_spot": "Bell Pepper Bacterial Spot", "Pepper,_bell___healthy": "Healthy Bell Pepper", "Potato___Early_blight": "Potato Early Blight",
        "Potato___Late_blight": "Potato Late Blight", "Potato___healthy": "Healthy Potato", "Raspberry___healthy": "Healthy Raspberry", "Soybean___healthy": "Healthy Soybean",
        "Squash___Powdery_mildew": "Squash Powdery Mildew", "Strawberry___Leaf_scorch": "Strawberry Leaf Scorch", "Strawberry___healthy": "Healthy Strawberry",
        "Tomato___Bacterial_spot": "Tomato Bacterial Spot", "Tomato___Early_blight": "Tomato Early Blight", "Tomato___Late_blight": "Tomato Late Blight",
        "Tomato___Leaf_Mold": "Tomato Leaf Mold", "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot", "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato Spider Mites",
        "Tomato___Target_Spot": "Tomato Target Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
        "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus", "Tomato___healthy": "Healthy Tomato"
    },
    "am": { 
        "Apple___Apple_scab": "የአፕል እከክ", "Apple___Black_rot": "የአፕል ጥቁር መበስበስ", "Apple___Cedar_apple_rust": "የአፕል የዝግባ አፕል ዝገት", "Apple___healthy": "ጤናማ አፕል",
        "Blueberry___healthy": "ጤናማ ብሉቤሪ", "Cherry_(including_sour)___Powdery_mildew": "የቼሪ የዱቄት አמחዳ", "Cherry_(including_sour)___healthy": "ጤናማ ቼሪ",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "የበቆሎ ሴርኮስፖራ ቅጠል ነጠብጣብ/ግራጫ ቅጠል ነጠብጣብ", "Corn_(maize)___Common_rust_": "የበቆሎ የተለመደ ዝገት",
        "Corn_(maize)___Northern_Leaf_Blight": "የበቆሎ የሰሜን ቅጠል ብላይት", "Corn_(maize)___healthy": "ጤናማ በቆሎ", "Grape___Black_rot": "የወይን ጥቁር መበስበስ",
        "Grape___Esca_(Black_Measles)": "የወይን ኤስካ (ጥቁር ኩፍኝ)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "የወይን ቅጠል ብላይት", "Grape___healthy": "ጤናማ ወይን",
        "Orange___Haunglongbing_(Citrus_greening)": "የብርቱካን ሁዋንግሎንግቢንግ (የሎሚ አረንጓዴ መሆን)", "Peach___Bacterial_spot": "የኮክ ባክቴሪያል ነጠብጣብ", "Peach___healthy": "ጤናማ ኮክ",
        "Pepper,_bell___Bacterial_spot": "የቃሪያ ባክቴሪያል ነጠብጣብ", "Pepper,_bell___healthy": "ጤናማ ቃሪያ", "Potato___Early_blight": "የድንች ቀደምት ብላይት",
        "Potato___Late_blight": "የድንች ዘግይቶ የሚመጣ ብላይት", "Potato___healthy": "ጤናማ ድንች", "Raspberry___healthy": "ጤናማ እንጆሪ", "Soybean___healthy": "ጤናማ አኩሪ አተር",
        "Squash___Powdery_mildew": "የዱባ የዱቄት አמחዳ", "Strawberry___Leaf_scorch": "የእንጆሪ ቅጠል ማቃጠል", "Strawberry___healthy": "ጤናማ እንጆሪ",
        "Tomato___Bacterial_spot": "የቲማቲም ባክቴሪያል ነጠብጣብ", "Tomato___Early_blight": "የቲማቲም ቀደምት ብላይት", "Tomato___Late_blight": "የቲማቲም ዘግይቶ የሚመጣ ብላይት",
        "Tomato___Leaf_Mold": "የቲማቲም ቅጠል ሻጋታ", "Tomato___Septoria_leaf_spot": "የቲማቲም ሴፕቶሪያ ቅጠል ነጠብጣብ", "Tomato___Spider_mites Two-spotted_spider_mite": "የቲማቲም የሸረሪት ሚይት",
        "Tomato___Target_Spot": "የቲማቲም የታለመ ነጠብጣብ", "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "የቲማቲም ቢጫ ቅጠል መጠምጠም ቫይረስ",
        "Tomato___Tomato_mosaic_virus": "የቲማቲም ሞዛይክ ቫይረስ", "Tomato___healthy": "ጤናማ ቲማቲም"
    },
    "ti": { 
        "Apple___Apple_scab": "ሕማም ስካብ ናይ ኣፕል", "Apple___Black_rot": "ጸሊም ምብላሽ ናይ ኣፕል", "Apple___Cedar_apple_rust": "ናይ ኣፕል ሕማም ዝገት ናይ ሲዳር ኣፕል", "Apple___healthy": "ጥዑይ ኣፕል",
        "Blueberry___healthy": "ጥዑይ ብሉቤሪ", "Cherry_(including_sour)___Powdery_mildew": "ናይ ቼሪ ሕማም ዱቄት ምብሳል", "Cherry_(including_sour)___healthy": "ጥዑይ ቼሪ",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "ናይ ዓዳጉራ ሕማም ቀጽሊ ሰርኮስፖራ/ግራጫ ነጠብጣብ ቀጽሊ", "Corn_(maize)___Common_rust_": "ናይ ዓዳጉራ ልሙድ ዝገት",
        "Corn_(maize)___Northern_Leaf_Blight": "ናይ ዓዳጉራ ሕማም ቀጽሊ ሰሜናዊ ብላይት", "Corn_(maize)___healthy": "ጥዑይ ዓዳጉራ", "Grape___Black_rot": "ጸሊም ምብላሽ ወይኒ",
        "Grape___Esca_(Black_Measles)": "ናይ ወይኒ ሕማም ኤስካ (ጸሊም ንፍሮ)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "ናይ ወይኒ ሕማም ቀጽሊ ብላይት", "Grape___healthy": "ጥዑይ ወይኒ",
        "Orange___Haunglongbing_(Citrus_greening)": "ናይ ኦሬንጅ ሕማም ሁዋንግሎንግቢንግ (ምልሓስ ሊባኖስ)", "Peach___Bacterial_spot": "ናይ ኮኽ ሕማም ባክቴሪያዊ ነጠብጣብ", "Peach___healthy": "ጥዑይ ኮኽ",
        "Pepper,_bell___Bacterial_spot": "ናይ በርበረ ሕማም ባክቴሪያዊ ነጠብጣብ", "Pepper,_bell___healthy": "ጥዑይ በርበረ", "Potato___Early_blight": "ናይ ድንሽ ሕማም ቀዳማይ ብላይት",
        "Potato___Late_blight": "ናይ ድንሽ ሕማም ዳሕረዋይ ብላይት", "Potato___healthy": "ጥዑይ ድንሽ", "Raspberry___healthy": "ጥዑይ ራስበሪ", "Soybean___healthy": "ጥዑይ ኣድሪ",
        "Squash___Powdery_mildew": "ናይ ዱባ ሕማም ዱቄት ምብሳል", "Strawberry___Leaf_scorch": "ናይ እንጆሪ ሕማም ምንዳድ ቀጽሊ", "Strawberry___healthy": "ጥዑይ እንጆሪ",
        "Tomato___Bacterial_spot": "ናይ ቲማቲም ሕማም ባክቴሪያዊ ነጠብጣብ", "Tomato___Early_blight": "ናይ ቲማቲም ሕማም ቀዳማይ ብላይት", "Tomato___Late_blight": "ናይ ቲማቲም ሕማም ዳሕረዋይ ብላይት",
        "Tomato___Leaf_Mold": "ናይ ቲማቲም ሕማም ሻጋታ ቀጽሊ", "Tomato___Septoria_leaf_spot": "ናይ ቲማቲም ሕማም ሴፕቶሪያ ነጠብጣብ ቀጽሊ", "Tomato___Spider_mites Two-spotted_spider_mite": "ናይ ቲማቲም ሕማም ሰራወር ሚይት",
        "Tomato___Target_Spot": "ናይ ቲማቲም ሕማም ዒላማ ነጠብጣብ", "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "ናይ ቲማቲም ቫይረስ ምጭብጫብ ቀጽሊ ብጫ ሕብሪ",
        "Tomato___Tomato_mosaic_virus": "ናይ ቲማቲም ቫይረስ ሞዛይክ", "Tomato___healthy": "ጥዑይ ቲማቲም"
    }
}

DISEASE_RECOMMENDATIONS = {
    "Apple___Apple_scab": {
        "en": {"description": "Fungal disease causing dark lesions on apple leaves and fruit.", "symptoms_list": ["Olive-green to black spots on leaves/fruit", "Distorted, prematurely falling leaves", "Corky fruit lesions"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Resistant varieties", "Sanitation: Rake/destroy fallen leaves/fruit", "Prune for air circulation", "Avoid wetting foliage"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides may be needed. Timing is critical. Consult local experts for product and schedule."},
        "am": {"description": "የአፕል ቅጠሎችን እና ፍራፍሬዎችን የሚያጠቃ የፈንገስ በሽታ ሲሆን ጥቁር ቁስሎችን ያስከትላል።", "symptoms_list": ["በቅጠሎች/ፍራፍሬዎች ላይ የወይራ አረንጓዴ እስከ ጥቁር ነጠብጣቦች", "የተዛቡ፣ ያለጊዜው የሚረግፉ ቅጠሎች", "የቡሽ መልክ ያላቸው የፍራፍሬ ቁስሎች"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ ዝርያዎች", "ንፅህና፡ የወደቁ ቅጠሎችን/ፍራፍሬዎችን ሰብስቦ ማጥፋት", "ለአየር ዝውውር መግረዝ", "ቅጠሎችን ከማርጠብ መቆጠብ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ፈንገስ መድኃኒቶች ሊያስፈልጉ ይችላሉ። ጊዜ አጠባበቅ ወሳኝ ነው። ለምርት እና የጊዜ ሰሌዳ የአካባቢ ባለሙያዎችን ያማክሩ።"},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ኣብ ቀጽሊ ኣፕልን ፍረን ጸሊም መልክዕ ዘለዎ ምዕንዛር ዘስዕብ።", "symptoms_list": ["ካብ ወይራ ዓይነት ቀጠልያ ክሳብ ጸሊም ዝሕብሩ ነጠብጣባት ኣብ ቀጽልን ፍረን", "ዝተጠወዩን ብዘይ ግዚኦም ዝረግፉ ቀጸልትን", "ከም ቡሽ ዝኣመሰሉ ምዕንዛራት ኣብ ፍረ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ ዝርኣያት ምምራጽ", "ጽሬት ምክያድ፡ ዝረገፉ ቀጸልትን ፍረታትን ምእካብን ምጥፋእን", "ንመዘዋወሪ ኣየር ምግላጽ", "ቀጸልቲ ከይረጥቡ ምክልኻል"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊ መድሃኒታት ኣድለይቲ ክኾኑ ይኽእሉ። ግዜ ኣጠቓቕማ ኣገዳሲ እዩ። ንውጽኢትን መደብ ግዜን ናይ ከባቢኹም ክኢላታት ኣማኽሩ።"}
    },
    "Apple___Black_rot": {
        "en": {"description": "Fungal disease causing leaf spots, fruit rot, and cankers.", "symptoms_list": ["'Frogeye' leaf spots", "Dark, enlarging fruit rot with concentric rings", "Mummified fruit", "Sunken cankers on branches"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Sanitation: Prune cankers, remove mummies/infected fruit", "Control insect damage", "Good air circulation"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides for other diseases may help. Specific products may be needed. Consult local experts."},
        "am": {"description": "የፈንገስ በሽታ ቅጠሎችን, የፍራፍሬ መበስበስ እና ካንሰሮችን ያስከትላል.", "symptoms_list": ["'የእንቁራሪት ዓይን' የሚመስሉ የቅጠል ነጠብጣቦች", "ጥቁር፣ እየሰፋ የሚሄድ የፍራፍሬ መበስበስ ከማዕከላዊ ቀለበቶች ጋር", "የደረቁ (ሙሚ የሆኑ) ፍሬዎች", "በቅርንጫፎች ላይ የሰመጡ ቁስሎች (ካንከሮች)"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["ንፅህና፡ ቁስሎችን (ካንከሮችን) መግረዝ፣ የደረቁ/የተበከሉ ፍራፍሬዎችን ማስወገድ", "የነፍሳት ጉዳትን መቆጣጠር", "ጥሩ የአየር ዝውውር"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ለሌሎች በሽታዎች የሚውሉ ፈንገስ መድኃኒቶች ሊረዱ ይችላሉ። የተወሰኑ ምርቶች ሊያስፈልጉ ይችላሉ። የአካባቢ ባለሙያዎችን ያማክሩ።"},
        "ti": {"description": "ሕማም ፋንጋስ ንነጠብጣብ ቆጽሊ፡ ምብስባስ ፍረታትን መንሽሮን ዘስዕብ.", "symptoms_list": ["ናይ 'ፍሮግኣይ' (ዒንቶዅራሪት) ዝመስሉ ነጠብጣባት ኣብ ቀጽሊ", "ጸሊምን እናገፍሐ ዝኸይድ ምብስባስ ፍረ ምስ ማእከላይ ቀለቤታት", "ዝደመሙ (ሙማይ ዝኾኑ) ፍረታት", "ኣብ ጨናፍር ዝጠሓሉ ቊስልታት (ካንከራት)"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ጽሬት፡ ቊስልታት (ካንከራት) ምፍላጥ፣ ዝደመሙ/ዝተበከሉ ፍረታት ምእላይ", "ጉድኣት ሸኻኺት ምቁጽጻር", "ጽቡቕ መዘዋወሪ ኣየር"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ንኻልኦት ሕማማት ዝውዕሉ ፈንገስ መከላኸሊ መድሃኒታት ክሕግዙ ይኽእሉ። ፍሉያት መፍረያት ከድልዩ ይኽእሉ። ናይ ከባቢኹም ክኢላታት ኣማኽሩ።"}
    },
    "Apple___Cedar_apple_rust": {
        "en": {"description": "Fungal disease requiring both apple/crabapple and cedar/juniper hosts to complete its lifecycle.", "symptoms_list": ["Bright orange/yellow spots on apple leaves, later with tiny black dots and tube-like structures underneath", "Fruit lesions possible", "On cedars: Brown galls that produce orange, gelatinous 'horns' in spring"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Resistant apple varieties", "Remove nearby cedar/juniper hosts if feasible (often impractical)", "Prune out cedar galls before they produce spores"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicide sprays on apples during infection periods (spring, when cedar galls are active). Consult local experts."},
        "am": {"description": "የፈንገስ በሽታ ሲሆን የህይወት ኡደቱን ለማጠናቀቅ አፕል/ክራብአፕል እና ሴዳር/ጁኒፐር አስተናጋጆችን ይፈልጋል።", "symptoms_list": ["በአፕል ቅጠሎች ላይ ደማቅ ብርቱካናማ/ቢጫ ነጠብጣቦች፣ በኋላ ላይ ትናንሽ ጥቁር ነጠብጣቦች እና ከስር የቱቦ መዋቅሮች ይታያሉ", "የፍራፍሬ ቁስሎች ሊኖሩ ይችላሉ", "በሴዳር ዛፎች ላይ፦ በፀደይ ወቅት ብርቱካናማ፣ ጄል የመሰሉ ' ቀንዶች' የሚያመነጩ ቡናማ እጢዎች"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ የአፕል ዝርያዎች", "ከተቻለ በአቅራቢያ ያሉ የሴዳር/ጁኒፐር አስተናጋጆችን ማስወገድ (ብዙውን ጊዜ ተግባራዊ አይሆንም)", "ስፖር ከማምረታቸው በፊት የሴዳር እጢዎችን መቁረጥ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "በበሽታው በሚተላለፍበት ወቅት (በፀደይ፣ የሴዳر እጢዎች ንቁ ሲሆኑ) በአፕል ላይ የፈንገስ መድኃኒት መርጨት። የአካባቢ ባለሙያዎችን ያማክሩ።"},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ንዑደት ህይወቱ ንምዝዛም ንኣፕል/ክራብኣፕልን ንሲዳር/ጁኒፐርን ከም ኣአንገድቲ ዝጥቀም።", "symptoms_list": ["ብሩህ ብርቱካናዊ/ብጫ ነጠብጣባት ኣብ ቀጽሊ ኣፕል፣ ድሒሩ ምስ ንኣሽቱ ጸለምቲ ነጠብጣባትን ከም ትቦ ዝኣመሰሉ ቅርጽታትን ኣብ ትሕቲኡ", "ቊስሊ ፍረታት ክኸውን ይኽእል", "ኣብ ሲዳር፡ ኣብ ጽድያ ብርቱካናዊ፡ ከም ጀል ዝበለ 'ቀርኒ' ዘፍርዩ ቡናዊ ዕንቊታት"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ ዝርኣያት ኣፕል", "ዝከኣል እንተኾይኑ፡ ኣብ ቀረባ ዝርከቡ ኣአንገድቲ ሲዳር/ጁኒፐር ምእላይ (መብዛሕትኡ ግዜ ተግባራዊ ኣይኮነን)", "ስፖር ከየፍረዩ ከለዉ ዕንቊታት ሲዳር ምፍላጥ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊ መረጻሕቲ ኣብ ኣፕል ኣብ እዋን ምትሕልላፍ ሕማም (ጽድያ፡ ዕንቊታት ሲዳር ንቑሓት ኣብ ዝኾንሉ እዋን)። ናይ ከባቢኹም ክኢላታት ኣማኽሩ።"}
    },
    "Apple___healthy": {
        "en": {"description": "The apple plant appears healthy and free of the targeted diseases.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Apple Care:", "cultural_control_list": ["Proper watering, fertilization, and pruning", "Monitor regularly for pests/diseases", "Good air circulation and sunlight"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Consider preventative sprays only if local disease pressure is high and after consulting experts."},
        "am": {"description": "የአፕል ተክል ጤናማ እና ከተጠቀሱት በሽታዎች የጸዳ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የአፕል እንክብካቤ:", "cultural_control_list": ["ትክክለኛ ውሃ ማጠጣት፣ ማዳበሪያ እና መግረዝ", "ለተባይ/በሽታዎች በየጊዜው መከታተል", "ጥሩ የአየር ዝውውር እና የፀሐይ ብርሃን"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "የመከላከያ መርጫዎችን ግምት ውስጥ ማስገባት ያለብዎት የአካባቢ የበሽታ ግፊት ከፍተኛ ከሆነ እና ከባለሙያዎች ጋር ከተማከሩ በኋላ ብቻ ነው።"},
        "ti": {"description": "እቲ ተኽሊ ኣፕል ጥዑይ ይመስል ካብቶም ዝተጠቕሱ ሕማማት ድማ ነጻ እዩ።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ኣፕል:", "cultural_control_list": ["ግቡእ ምስታይ ማይ፣ ምዳበሪያን ምፍላጥን", "ንተሃሳስን ሕማማትን ብቐጻሊ ምቁጽጻር", "ጽቡቕ መዘዋወሪ ኣየርን ብርሃን ጸሓይን"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "መከላኸሊ መረጻሕቲ ግምት ውስጥ ክኣትዉ ዘለዎም፡ ናይ ከባቢ ጸቕጢ ሕማም ልዑል እንተኾይኑን ምስ ክኢላታት ድሕሪ ምምኻርን ጥራይ እዩ።"}
    },
    "Blueberry___healthy": {
        "en": {"description": "The blueberry plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Blueberry Care:", "cultural_control_list": ["Acidic, well-drained soil", "Proper watering and mulching", "Annual pruning", "Monitor for pests/diseases"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on cultural practices; use chemicals only if necessary and after expert consultation."},
        "am": {"description": "የብሉቤሪ ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የብሉቤሪ እንክብካቤ:", "cultural_control_list": ["አሲዳማ፣ ውሃ በደንብ የሚያሳልፍ አፈር", "ትክክለኛ ውሃ ማጠጣት እና መሸፈኛ ማድረግ", "ዓመታዊ መግረዝ", "ለተባይ/በሽታዎች መከታተል"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በባህላዊ ልምዶች ላይ ያተኩሩ፤ ኬሚካሎችን አስፈላጊ ከሆነ እና ከባለሙያ ምክክር በኋላ ብቻ ይጠቀሙ።"},
        "ti": {"description": "እቲ ተኽሊ ብሉቤሪ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ብሉቤሪ:", "cultural_control_list": ["ኣሲዳዊ፡ ማይ ጽቡቕ ዘንጠብጥብ ሓመድ", "ግቡእ ምስታይ ማይን ምጉስጓስን", "ዓመታዊ ምፍላጥ", "ንተሃሳስን ሕማማትን ምቁጽጻር"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ባህላዊ ተግባራት ኣተኵሩ፤ ኬሚካላት ኣድላዪ እንተኾይኑን ምስ ክኢላ ድሕሪ ምምኻርን ጥራይ ተጠቐሙ።"}
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "en": {"description": "Fungal disease causing a white, powdery coating on cherry leaves and fruit.", "symptoms_list": ["White powdery patches on leaves (upper or lower surface), shoots, and sometimes fruit", "Distorted leaf growth", "Premature leaf drop in severe cases"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Prune for good air circulation", "Avoid excessive nitrogen fertilization (promotes susceptible new growth)", "Remove and destroy infected plant parts if practical", "Plant in sunny locations"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides (e.g., sulfur, potassium bicarbonate, or systemic options) can be effective. Apply at first sign of disease. Consult local experts."},
        "am": {"description": "በቼሪ ቅጠሎች እና ፍራፍሬዎች ላይ ነጭ፣ የዱቄት ሽፋን የሚያስከትል የፈንገስ በሽታ።", "symptoms_list": ["በቅጠሎች (የላይኛው ወይም የታችኛው ገጽ)፣ ቀንበጦች እና አንዳንዴም ፍራፍሬዎች ላይ ነጭ የዱቄት መልክ ያላቸው ምልክቶች", "የተዛባ የቅጠል እድገት", "በከባድ ሁኔታዎች ቅጠሎች ያለጊዜው መውደቅ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["ለጥሩ የአየር ዝውውር መግረዝ", "ከመጠን በላይ የናይትሮጅን ማዳበሪያን ማስወገድ (ለበሽታ ተጋላጭ የሆነ አዲስ እድገትን ያበረታታል)", "ከተቻለ የተበከሉ የተክል ክፍሎችን ማስወገድ እና ማጥፋት", "ፀሐያማ በሆኑ ቦታዎች መትከል"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ፈንገስ መድኃኒቶች (ለምሳሌ፦ ሰልፈር፣ ፖታሺየም ባይካርቦኔት ወይም ሥርዓታዊ አማራጮች) ውጤታማ ሊሆኑ ይችላሉ። የበሽታው የመጀመሪያ ምልክት ሲታይ ይተግብሩ። የአካባቢ ባለሙያዎችን ያማክሩ።"},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ኣብ ቀጽሊ ቼሪን ፍረታትን ጻዕዳ፡ ከም ዱቄት ዝበለ መሸፈኒ ዘስዕብ።", "symptoms_list": ["ጻዕዳ ከም ዱቄት ዝበለ ነጠብጣባት ኣብ ቀጸልቲ (ላዕለዋይ ወይ ታሕተዋይ ገጽ)፡ ጠጥዕምን ሓሓሊፉ ኣብ ፍረታትን", "ዝተጓነየ ዕቤት ቀጽሊ", "ኣብ ከቢድ ኩነታት፡ ቀጽሊ ብዘይ ግዚኡ ምርጋፍ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ንጽቡቕ መዘዋወሪ ኣየር ምፍላጥ", "ልዑል ምዳበሪያ ናይትሮጅን ምውጋድ (ንሕማም ተቓላዒ ዝኾነ ሓድሽ ዕቤት የሀዊኽ)", "ዝከኣል እንተኾይኑ፡ ዝተበከሉ ክፋላት ተኽሊ ምእላይን ምጥፋእን", "ኣብ ጸሓያዊ ቦታታት ምትካል"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊ መድሃኒታት (ከም ኣብነት፡ ሰልፈር፡ ፖታሽየም ባይካርቦኔት፡ ወይ ስነ-ስርዓታዊ ኣማራጺታት) ውጽኢታውያን ክኾኑ ይኽእሉ። ኣብ ቀዳማይ ምልክት ሕማም ተጠቐሙ። ናይ ከባቢኹም ክኢላታት ኣማኽሩ።"}
    },
    "Cherry_(including_sour)___healthy": {
        "en": {"description": "The cherry plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Cherry Care:", "cultural_control_list": ["Well-drained soil", "Proper watering", "Annual pruning to maintain shape and air circulation", "Monitor for pests (e.g., borers, aphids) and diseases"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on cultural practices. Dormant oil sprays can help with some pests. Consult experts for disease prevention."},
        "am": {"description": "የቼሪ ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የቼሪ እንክብካቤ:", "cultural_control_list": ["ውሃ በደንብ የሚያሳልፍ አፈር", "ትክክለኛ ውሃ ማጠጣት", "ቅርፅን እና የአየር ዝውውርን ለመጠበቅ ዓመታዊ መግረዝ", "ለተባይ (ለምሳሌ፦ ቦረር፣ አፊድ) እና በሽታዎች መከታተል"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በባህላዊ ልምዶች ላይ ያተኩሩ። የዶርማንት ዘይት መርጫዎች ከአንዳንድ ተባዮች ሊረዱ ይችላሉ። ለበሽታ መከላከል ባለሙያዎችን ያማክሩ።"},
        "ti": {"description": "እቲ ተኽሊ ቼሪ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ቼሪ:", "cultural_control_list": ["ማይ ጽቡቕ ዘንጠብጥብ ሓመድ", "ግቡእ ምስታይ ማይ", "ቅርጽን መዘዋወሪ ኣየርን ንምዕቃብ ዓመታዊ ምፍላጥ", "ንተሃሳስ (ከም ኣብነት፡ ቦረር፡ ኣፊድ)ን ሕማማትን ምቁጽጻር"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ባህላዊ ተግባራት ኣተኵሩ። ኣብ እዋን ዕረፍቲ ዝግበር መረጻሕቲ ዘይቲ ንገሊኦም ተሃሳስ ክሕግዙ ይኽእሉ። ንምክልኻል ሕማም ምስ ክኢላታት ተማኸሩ።"}
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "en": {"description": "Fungal disease causing rectangular, grayish lesions on corn leaves, significantly impacting yield.", "symptoms_list": ["Long, narrow, rectangular lesions (1-4 inches) parallel to leaf veins", "Initially tan, turning grayish-brown", "Lesions can merge, blighting large areas of leaves"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Resistant/tolerant hybrids (most effective)", "Crop rotation (at least 1 year away from corn)", "Tillage to bury infected residue", "Manage irrigation to avoid prolonged leaf wetness"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Foliar fungicides can be effective, especially if applied around tasseling when disease is present on lower leaves. Scout fields and consider economic thresholds. Consult local advisors."},
        "am": {"description": "በቆሎ ቅጠሎች ላይ አራት መአዘን፣ ግራጫማ ቁስሎችን የሚያስከትል የፈንገስ በሽታ ሲሆን ምርትን በእጅጉ ይጎዳል።", "symptoms_list": ["ከቅጠል ሥሮች ጋር ትይዩ የሆኑ ረጅም፣ ጠባብ፣ አራት መአዘን ቁስሎች (1-4 ኢንች)", "መጀመሪያ ላይ ፈዛዛ ቡኒ፣ ወደ ግራጫ-ቡኒነት የሚቀየር", "ቁስሎች ሊዋሃዱ ይችላሉ፣ ትላልቅ የቅጠል ቦታዎችን ያበላሻሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ/የሚችሉ ዝርያዎች (በጣም ውጤታማ)", "የሰብል ማሽከርከር (ከቆሎ ቢያንስ 1 አመት የራቀ)", "የተበከሉ ቀሪዎችን ለመቅበር ማረስ", "የቅጠል እርጥበትን ለረጅም ጊዜ ለማስወገድ መስኖን ማስተዳደር"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "የቅጠል ፈንገስ መድኃኒቶች ውጤታማ ሊሆኑ ይችላሉ፣ በተለይም በሽታው በታችኛው ቅጠሎች ላይ በሚገኝበት ጊዜ በአበባ መውጫ አካባቢ ከተተገበሩ። ማሳዎችን ይቃኙ እና ኢኮኖሚያዊ ገደቦችን ግምት ውስጥ ያስገቡ። የአካባቢ አማካሪዎችን ያማክሩ።"},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ኣብ ቀጽሊ ዓዳጉራ ኣርባዕተ ኩርናዓዊ፡ ግራጫዊ ቊስሊ ዘስዕብ፡ ንፍርያት ብዓቢኡ ዝጸሉ።", "symptoms_list": ["ነዊሕ፡ ቀጢን፡ ኣርባዕተ ኩርናዓዊ ቊስሊ (1-4 ኢንች) ምስ ሰራውር ቀጽሊ ዝመሳሰል", "መጀመርታ ሃመዳዊ ሕብሪ፡ ናብ ግራጫ-ቡናዊ ዝቕየር", "ቊስልታት ክሓብሩ ይኽእሉ፡ ዓበይቲ ክፋላት ቀጽሊ ከኣ የንውጹ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ/ትዕግስተኛታት ዝርኣያት (እቲ ዝበለጸ ውጽኢታዊ)", "ክቢ ዘራእቲ (እንተወሓደ ን1 ዓመት ካብ ዓዳጉራ ዝርሕቕ)", "ዝተበከለ ተረፍ ንምቕባር ምሕራስ", "ነዊሕ ግዜ ዝጸንሕ ጥልቀት ቀጽሊ ንምክልኻል መስኖ ምምሕዳር"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊታት ቀጽሊ ውጽኢታውያን ክኾኑ ይኽእሉ፡ ብፍላይ ኣብ እዋን ምዕምባብ፡ ሕማም ኣብ ታሕተዎት ቀጸልቲ ኣብ ዝርከበሉ እዋን እንተተጠቒሞም። ግራውቲ መርምሩን ቁጠባዊ ደረታት ግምት ኣብ ግምት ኣእትዉን። ናይ ከባቢኹም ኣማኸርቲ ተወከሱ።"}
    },
    "Corn_(maize)___Common_rust_": {
        "en": {"description": "Fungal disease characterized by reddish-brown, powdery pustules on corn leaves.", "symptoms_list": ["Cinnamon-brown, oval to elongated powdery pustules on both leaf surfaces", "Pustules may be surrounded by a yellow halo", "Severe infections can cause leaf yellowing/death"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Resistant/tolerant hybrids (primary strategy)", "Early planting may help avoid peak spore loads", "Good air circulation"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Foliar fungicides can be used, especially on susceptible hybrids or for seed production, if applied early.Consult local advisors for thresholds."},
        "am": {"description": "በቆሎ ቅጠሎች ላይ ቀይ-ቡኒ፣ የዱቄት መልክ ያላቸው እብጠቶችን የሚያሳይ የፈንገስ በሽታ።", "symptoms_list": ["በሁለቱም የቅጠል ገጾች ላይ ቀረፋ-ቡኒ፣ ሞላላ እስከ ረዣዥም የዱቄት እብጠቶች", "እብጠቶች በቢጫ ቀለበት ሊከበቡ ይችላሉ", "ከባድ ኢንፌክሽኖች የቅጠል ቢጫነት/ሞት ሊያስከትሉ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ/የሚችሉ ዝርያዎች (ዋና ስትራቴጂ)", "ቀደም ብሎ መትከል ከፍተኛ የስፖሮ ጭነትን ለማስወገድ ሊረዳ ይችላል", "ጥሩ የአየር ዝውውር"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "የቅጠል ፈንገስ መድኃኒቶች ጥቅም ላይ ሊውሉ ይችላሉ፣ በተለይም ለበሽታ ተጋላጭ በሆኑ ዝርያዎች ወይም ለዘር ምርት፣ ቀደም ብለው ከተተገበሩ። ለገደቦች የአካባቢ አማካሪዎችን ያማክሩ።"},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ቀይሕ-ቡናዊ፡ ከም ዱቄት ዝበለ ቁስሊ ኣብ ቀጽሊ ዓዳጉራ ዘርኢ።", "symptoms_list": ["ከም ቀረፋ ዝሕብሩ፡ ሞላላ ወይ ነዊሕ ዝበሉ ከም ዱቄት ዝኾኑ ቁስልታት ኣብ ክልቲኡ ገጻት ቀጽሊ", "እቶም ቁስልታት ብብጫ ቀለቤት ክኽበቡ ይኽእሉ", "ብርቱዕ ምትሕልላፍ ሕማም ንቀጽሊ ናብ ብጫነት/ሞት ከምርሖ ይኽእል"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ/ትዕግስተኛታት ዝርኣያት (ቀንዲ ስትራተጂ)", "ኣቐዲምካ ምዝራእ ንልዑል ጽዕነት ስፖር ንምክልኻል ክሕግዝ ይኽእል", "ጽቡቕ መዘዋወሪ ኣየር"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊታት ቀጽሊ ክጥቀሙ ይኽእሉ፡ ብፍላይ ኣብቶም ተቓለዕቲ ዝርኣያት ወይ ንምፍራይ ዘርኢ፡ ኣቐዲሞም እንተተጠቒሞም። ንደረጃታት ምስ ናይ ከባቢኹም ኣማኸርቲ ተወከሱ።"}
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "en": {"description": "Fungal disease causing long, elliptical, grayish-green to tan lesions on corn leaves.", "symptoms_list": ["Large (1-6 inches), cigar-shaped, tan or grayish lesions on leaves", "Lesions can coalesce, blighting entire leaves", "Characteristic 'dirty' appearance due to fungal sporulation within lesions"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Resistant/tolerant hybrids (most effective)", "Crop rotation", "Tillage to bury residue", "Avoid planting corn near previously infected fields"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Foliar fungicides can be effective, particularly if applied preventatively or at early onset on susceptible hybrids. Consult local advisors.", "further_info_link": ""},
        "am": {"description": "በቆሎ ቅጠሎች ላይ ረጅም፣ ሞላላ፣ ግራጫ-አረንጓዴ እስከ ፈዛዛ ቡኒ ቁስሎችን የሚያስከትል የፈንገስ በሽታ።", "symptoms_list": ["ትልቅ (1-6 ኢንች)፣ ሲጋራ የመሰለ ቅርፅ ያላቸው፣ ፈዛዛ ቡኒ ወይም ግራጫማ ቁስሎች በቅጠሎች ላይ", "ቁስሎች ሊዋሃዱ ይችላሉ፣ ሙሉ ቅጠሎችን ያበላሻሉ", "በቁስሎች ውስጥ ባለው የፈንገስ ስፖሮሌሽን ምክንያት የ'ቆሻሻ' መልክ መታየት"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ/የሚችሉ ዝርያዎች (በጣም ውጤታማ)", "የሰብል ማሽከርከር", "ቀሪዎችን ለመቅበር ማረስ", "ቀደም ሲል በተበከሉ ማሳዎች አቅራቢያ በቆሎ ከመትከል መቆጠብ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "የቅጠል ፈንገስ መድኃኒቶች ውጤታማ ሊሆኑ ይችላሉ፣ በተለይም ለበሽታ ተጋላጭ በሆኑ ዝርያዎች ላይ በመከላከል ወይም ገና ሲጀምር ከተተገበሩ። የአካባቢ አማካሪዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ኣብ ቀጽሊ ዓዳጉራ ነዊሕ፡ ሞላላ፡ ግራጫ-ቀጠልያ ወይ ሃመዳዊ ቊስሊ ዘስዕብ።", "symptoms_list": ["ዓቢ (1-6 ኢንች)፡ ከም ሲጋራ ዝቅርጹ፡ ሃመዳዊ ወይ ግራጫዊ ቊስሊ ኣብ ቀጸልቲ", "ቊስልታት ክሓብሩ ይኽእሉ፡ ንብምሉኦም ቀጸልቲ ከኣ የንውጹ", "ብሰንኪ ኣብ ውሽጢ ቊስልታት ዝፍጠር ስፖር ፈንገስ፡ ፍሉይ 'ርስሓት' ዝመስል ትርኢት"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ/ትዕግስተኛታት ዝርኣያት (እቲ ዝበለጸ ውጽኢታዊ)", "ክቢ ዘራእቲ", "ተረፍ ንምቕባር ምሕራስ", "ኣብ ቀረባ ናይ ዝተበከሉ ግራውቲ፡ ዓዳጉራ ካብ ምዝራእ ምክልኻል"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊታት ቀጽሊ ውጽኢታውያን ክኾኑ ይኽእሉ፡ ብፍላይ ኣብቶም ተቓለዕቲ ዝርኣያት፡ ከም መከላኸሊ ወይ ኣብ መጀመርታ ሕማም እንተተጠቒሞም። ናይ ከባቢኹም ኣማኸርቲ ተወከሱ።", "further_info_link": ""}
    },
    "Corn_(maize)___healthy":{
        "en": {"description": "The corn plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Corn Care:", "cultural_control_list": ["Proper soil fertility and pH", "Adequate spacing for sunlight and air", "Timely planting", "Weed and insect control", "Appropriate watering"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on good agronomic practices and hybrid selection. Use chemicals as needed based on scouting and expert advice.", "further_info_link": ""},
        "am": {"description": "የበቆሎ ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የበቆሎ እንክብካቤ:", "cultural_control_list": ["ትክክለኛ የአፈር ለምነት እና ፒኤች", "ለፀሐይ ብርሃን እና አየር በቂ ክፍተት", "ወቅታዊ መትከል", "የአረም እና የተባይ መቆጣጠሪያ", "ተገቢ ውሃ ማጠጣት"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በጥሩ የግብርና ልምዶች እና በዘር ምርጫ ላይ ያተኩሩ። በቅኝት እና በባለሙያ ምክር መሰረት እንደአስፈላጊነቱ ኬሚካሎችን ይጠቀሙ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ ዓዳጉራ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ዓዳጉራ:", "cultural_control_list": ["ግቡእ ልምዓት ሓመድን ፒኤችን", "ንብርሃን ጸሓይን ኣየርን ዝኸውን በቂ ቦታ ምትካል", "ኣብ ግዚኡ ምዝራእ", "ምቁጽጻር ኣእዋምን ተሃሳስን", "ግቡእ ምስታይ ማይ"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ጽቡቕ ኣገባባት ሕርሻን ምርጫ ዝርኣያትን ኣተኵሩ። ከም ኣድላይነቱ፡ ኣብ ምምርማርን ምኽሪ ክኢላታትን ተመርኲስኩም ኬሚካላት ተጠቐሙ።", "further_info_link": ""}
    },
    "Grape___Black_rot": {
        "en": {"description": "A serious fungal disease of grapes, affecting leaves, shoots, and especially fruit.", "symptoms_list": ["Small, reddish-brown circular spots on leaves that enlarge, with dark borders and tiny black fungal fruiting bodies (pycnidia) in the center", "Fruit infections start as light spots, then rot, turn black, shrivel, and become hard ('mummies')"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Sanitation: Remove and destroy mummified fruit and infected canes during dormant pruning", "Improve air circulation through pruning and canopy management", "Weed control"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "A rigorous fungicide spray program is often essential, starting early in the season. Consult local grape advisors for product selection and timing."},
        "am": {"description": "ከባድ የወይን የፈንገስ በሽታ ሲሆን ቅጠሎችን፣ ቀንበጦችን እና በተለይም ፍራፍሬዎችን ያጠቃል።", "symptoms_list": ["በቅጠሎች ላይ ትናንሽ፣ ቀይ-ቡኒ ክብ ነጠብጣቦች እየሰፉ የሚሄዱ፣ ጥቁር ድንበር እና በማዕከላቸው ትናንሽ ጥቁር የፈንገስ ፍሬያማ አካላት (ፒክኒዲያ) ያላቸው", "የፍራፍሬ ኢንፌክሽኖች እንደ ፈዛዛ ነጠብጣብ ይጀምራሉ፣ ከዚያም ይበሰብሳሉ፣ ይጠቁራሉ፣ ይሸበሸባሉ፣ እና ይጠነክራሉ ('ሙሚዎች')"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["ንፅህና፦ በእንቅልፍ ወቅት በሚደረግ መግረዝ የደረቁ ፍራፍሬዎችን እና የተበከሉ ቀንበጦችን ማስወገድ እና ማጥፋት", "በመግረዝ እና በቅጠላ ቅጠል አያያዝ የአየር ዝውውርን ማሻሻል", "የአረም መቆጣጠሪያ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ጠንካራ የፈንገስ መድኃኒት የመርጨት ፕሮግራም ብዙውን ጊዜ አስፈላጊ ነው፣ ከወቅቱ መጀመሪያ ጀምሮ። ለምርት ምርጫ እና ጊዜ አጠባበቅ የአካባቢ የወይን አማካሪዎችን ያማክሩ።"},
        "ti": {"description": "ከቢድ ፈንገሳዊ ሕማም ወይኒ፡ ንቀጸልቲ፡ ጠጥዒ፡ ብፍላይ ከኣ ንፍረታት ዝጎድእ።", "symptoms_list": ["ኣብ ቀጸልቲ ንኣሽቱ፡ ቀይሕ-ቡናዊ ክብ ነጠብጣባት እናገፍሑ ዝኸዱ፡ ጸሊም ደረት ዘለዎምን ኣብ ማእከሎም ንኣሽቱ ጸለምቲ ፈንገሳዊ ኣካላት ፍረ (ፒክኒድያ) ዘለዎምን", "ምትሕልላፍ ሕማም ፍረታት ከም ፈኲስ ነጠብጣብ ይጅምር፡ ድሕሪኡ ይበስብስ፡ ይጸልም፡ ይጭምድድ፡ ይተርር ('ሙማይ')"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ጽሬት፡ ኣብ እዋን ዕረፍቲ ዝግበር ምፍላጥ፡ ዝደመሙ ፍረታትን ዝተበከሉ ጨናፍርን ምእላይን ምጥፋእን", "ብምፍላጥን ምሕደራ ቆጽሊን መዘዋወሪ ኣየር ምምሕያሽ", "ምቁጽጻር ኣእዋም"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ጽኑዕ ናይ ፈንገስ መከላኸሊ መረጻሕቲ ፕሮግራም መብዛሕትኡ ግዜ ኣድላዪ እዩ፡ ካብ መጀመርታ ወቕቲ ጀሚሩ። ንምርጫ ፍርያትን ግዜን ምስ ናይ ከባቢኹም ኣማኸርቲ ወይኒ ተወከሱ።"}
    },
    "Grape___Esca_(Black_Measles)": {
        "en": {"description": "A complex fungal trunk disease of grapevines, often leading to decline and death.", "symptoms_list": ["Leaf symptoms: Interveinal chlorosis or reddening ('tiger stripes'), drying from margins inward", "Fruit symptoms: Small, dark spots ('measles') on berries", "Wood symptoms: Dark streaking or wedge-shaped necrosis in cross-sections of trunk/cordons"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Avoid wounding vines, especially during wet weather", "Protect large pruning wounds with a sealant", "Remove and destroy severely infected vines", "Good vineyard sanitation", "Delayed pruning can sometimes help"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Currently, no highly effective chemical treatments are available to cure Esca. Focus is on prevention and managing infected vines. Some trunk injection products are experimental. Consult specialists."},
        "am": {"description": "የወይን ተክል ግንድ ላይ የሚከሰት ውስብስብ የፈንገስ በሽታ ሲሆን ብዙውን ጊዜ ወደ ተክሉ መዳከም እና ሞት ይመራል።", "symptoms_list": ["የቅጠል ምልክቶች፦ በደም ሥሮች መካከል ቢጫ መሆን ወይም መቅላት ('የነብር መስመር')፣ ከዳር ወደ ውስጥ መድረቅ", "የፍራፍሬ ምልክቶች፦ በቤሪ ፍሬዎች ላይ ትናንሽ፣ ጥቁር ነጠብጣቦች ('ኩፍኝ')", "የእንጨት ምልክቶች፦ በግንድ/ቅርንጫፍ መስቀለኛ ክፍል ላይ ጥቁር መስመሮች ወይም የሽብልቅ ቅርጽ ያለው ኒክሮሲስ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["በተለይም በእርጥብ የአየር ጠባይ ወቅት የወይን ተክሎችን ከመጉዳት መቆጠብ", "ትላልቅ የመግረዝ ቁስሎችን በማሸጊያ መጠበቅ", "በጠና የተበከሉ የወይን ተክሎችን ማስወገድ እና ማጥፋት", "ጥሩ የወይን እርሻ ንፅህና", "ዘግይቶ መግረዝ አንዳንድ ጊዜ ሊረዳ ይችላል"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "በአሁኑ ጊዜ ኤስካን ለመፈወስ ምንም ውጤታማ የኬሚካል ሕክምናዎች የሉም። ትኩረቱ በመከላከል እና የተበከሉ የወይን ተክሎችን በማስተዳደር ላይ ነው። አንዳንድ የግንድ መርፌ ምርቶች በሙከራ ላይ ናቸው። ልዩ ባለሙያዎችን ያማክሩ።"},
        "ti": {"description": "ውስብስብ ፈንገሳዊ ሕማም ግንዲ ወይኒ፡ መብዛሕትኡ ግዜ ናብ ምዝላዕን ሞትን ዘምርሕ።", "symptoms_list": ["ምልክታት ቀጽሊ፡ ኣብ መንጎ ሰራውር ብጫ ምዃን ወይ ምቕያሕ ('መስመር ነብሪ')፡ ካብ ወሰን ናብ ውሽጢ ምድራቕ", "ምልክታት ፍረ፡ ኣብ ፍረታት ንኣሽቱ ጸለምቲ ነጠብጣባት ('ንፍሮ')", "ምልክታት ዕንጨይቲ፡ ኣብ መስቀላዊ ክፍልታት ግንዲ/ጨናፍር ጸሊም መስመር ወይ ከም ሽብልቅ ዝበለ ኒክሮሲስ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ብፍላይ ኣብ ጥሉል ኩነታት ኣየር ንተኽሊ ወይኒ ካብ ምጕዳእ ምክልኻል", "ዓበይቲ ናይ ምፍላጥ ቊስልታት ብመዕጸዊ ምዕቋብ", "ብርቱዕ ዝተበከሉ ተኽልታት ወይኒ ምእላይን ምጥፋእን", "ጽቡቕ ጽሬት ግራት ወይኒ", "ዘግዩ ምፍላጥ ሓሓሊፉ ክሕግዝ ይኽእል"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ሕጂ፡ ንኤስካ ንምፍዋስ ዝኾነ ብሉጽ ውጽኢታዊ ኬሚካላዊ ኣገባብ የለን። ትኹረት ኣብ ምክልኻልን ምሕደራ ዝተበከሉ ተኽልታት ወይንን እዩ። ገለ ናይ ግንዲ መውጋእቲ ፍርያት ኣብ ፈተነ እዮም። ምስ ክኢላታት ተማኸሩ።"}
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "en": {"description": "Fungal disease primarily affecting grape leaves, usually later in the season.", "symptoms_list": ["Irregular, dark brown to black lesions on leaves, often with a yellowish halo", "Underside of lesions may have a sooty or olive-green appearance due to fungal growth", "Severe infections can lead to defoliation"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Improve air circulation through canopy management", "Sanitation: Rake and destroy fallen leaves", "Avoid overhead irrigation"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides applied for other grape diseases (like black rot or downy mildew) often provide control. Specific applications may not be needed unless disease is severe. Consult local advisors.", "further_info_link": ""},
        "am": {"description": "በዋናነት የወይን ቅጠሎችን የሚያጠቃ የፈንገስ በሽታ ሲሆን ብዙውን ጊዜ በወቅቱ መጨረሻ ላይ ይከሰታል።", "symptoms_list": ["በቅጠሎች ላይ መደበኛ ያልሆኑ፣ ጥቁር ቡኒ እስከ ጥቁር ቁስሎች፣ ብዙውን ጊዜ ቢጫ ቀለበት ያላቸው", "በፈንገስ እድገት ምክንያት የቁስሎቹ የታችኛው ክፍል ጥቀርሻ ወይም የወይራ-አረንጓዴ መልክ ሊኖረው ይችላል", "ከባድ ኢንፌክሽኖች ቅጠሎችን ሊያረግፉ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["በቅጠላ ቅጠል አያያዝ የአየር ዝውውርን ማሻሻል", "ንፅህና፦ የወደቁ ቅጠሎችን መሰብሰብ እና ማጥፋት", "ከላይ የሚደረግ መስኖን ማስወገድ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ለሌሎች የወይን በሽታዎች (እንደ ጥቁር መበስበስ ወይም ዳውኒ ሚልዲው) የሚተገበሩ ፈንገስ መድኃኒቶች ብዙውን ጊዜ ቁጥጥር ይሰጣሉ። በሽታው ከባድ ካልሆነ በስተቀር የተለዩ ትግበራዎች ላያስፈልጉ ይችላሉ። የአካባቢ አማካሪዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ብቐንዱ ንቀጽሊ ወይኒ ዝጎድእ፡ መብዛሕትኡ ግዜ ኣብ መወዳእታ ወቕቲ።", "symptoms_list": ["ዘይስሩዕ፡ ጸሊም ቡናዊ ክሳብ ጸሊም ዝሕብሩ ቊስልታት ኣብ ቀጸልቲ፡ መብዛሕትኡ ግዜ ምስ ብጫ ቀለቤት", "ታሕተዋይ ገጽ ቊስልታት ብሰንኪ ዕቤት ፈንገስ ከም ሓሙኽሽታይ ወይ ወይራ-ቀጠልያ ዝመስል ትርኢት ክህልዎ ይኽእል", "ብርቱዕ ምትሕልላፍ ሕማም ናብ ምርጋፍ ቀጽሊ ከምርሕ ይኽእል"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ብምሕደራ ቆጽሊ መዘዋወሪ ኣየር ምምሕያሽ", "ጽሬት፡ ዝረገፉ ቀጸልቲ ምእካብን ምጥፋእን", "ካብ ላዕሊ ዝግበር መስኖ ምውጋድ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ንኻልኦት ሕማማት ወይኒ (ከም ጸሊም ምብስባስ ወይ ዳውኒ ሚልድዩ) ዝውዕሉ ፈንገስ መከላኸሊ መድሃኒታት መብዛሕትኡ ግዜ ምቁጽጻር ይህቡ። ሕማም ከቢድ እንተዘይኮይኑ ፍሉያት ኣጠቓቕማ ኣየድልዩን ይኽእሉ። ናይ ከባቢኹም ኣማኸርቲ ተወከሱ።", "further_info_link": ""}
    },
    "Grape___healthy": {
        "en": {"description": "The grape plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Grape Care:", "cultural_control_list": ["Proper site selection (sunlight, drainage)", "Appropriate trellis system", "Annual pruning", "Soil management and fertilization", "Pest and weed control"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on cultural practices. Implement a preventative spray program if common diseases are prevalent in your area, based on expert advice.", "further_info_link": ""},
        "am": {"description": "የወይን ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የወይን እንክብካቤ:", "cultural_control_list": ["ትክክለኛ የቦታ ምርጫ (የፀሐይ ብርሃን፣ የውሃ ፍሳሽ)", "ተገቢ የድጋፍ ስርዓት", "ዓመታዊ መግረዝ", "የአፈር አያያዝ እና ማዳበሪያ", "የተባይ እና የአረም መቆጣጠሪያ"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በባህላዊ ልምዶች ላይ ያተኩሩ። የተለመዱ በሽታዎች በአካባቢዎ የተስፋፉ ከሆኑ በባለሙያ ምክር ላይ በመመርኮዝ የመከላከያ የመርጨት ፕሮግራም ይተግብሩ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ ወይኒ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ወይኒ:", "cultural_control_list": ["ግቡእ ምርጫ ቦታ (ብርሃን ጸሓይ፡ ምጽፋፍ ማይ)", "ዝግባእ ስርዓተ ምድጋፍ", "ዓመታዊ ምፍላጥ", "ምሕደራ ሓመድን ምዳበሪያን", "ምቁጽጻር ተሃሳስን ኣእዋምን"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ባህላዊ ተግባራት ኣተኵሩ። ልሙዳት ሕማማት ኣብ ከባቢኹም ዝተራእዩ እንተኾይኖም፡ ኣብ ምኽሪ ክኢላታት ተመርኲስኩም መከላኸሊ ናይ መረጻሕቲ ፕሮግራም ተግበሩ።", "further_info_link": ""}
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "en": {"description": "A devastating bacterial disease of citrus trees, spread by the Asian citrus psyllid. Currently incurable.", "symptoms_list": ["Yellowing of leaves, often blotchy and asymmetrical (unlike nutrient deficiencies)", "Small, upright leaves", "Misshapen, bitter, small fruit that may remain green at one end", "Twig dieback, eventual tree death"], "cultural_control_header": "Management (No Cure):", "cultural_control_list": ["Strict quarantine measures to prevent spread", "Control of the Asian citrus psyllid vector (insecticide applications, biological control)", "Removal and destruction of infected trees to reduce inoculum source", "Planting certified disease-free trees"], "chemical_control_header": "Vector Control:", "chemical_control_text": "Regular insecticide applications are crucial for psyllid control in affected areas. Consult local citrus advisors and follow regulations strictly.", "further_info_link": ""},
        "am": {"description": "የሎሚ ዛፎችን የሚያጠፋ ከባድ የባክቴሪያ በሽታ ሲሆን በእስያ የሎሚ ፕሲሊድ ይተላለፋል። በአሁኑ ጊዜ የማይድን ነው።", "symptoms_list": ["የቅጠሎች ቢጫ መሆን፣ ብዙውን ጊዜ ያልተስተካከለ እና ያልተመጣጠነ (ከምግብ እጥረት በተለየ)", "ትናንሽ፣ ቀጥ ያሉ ቅጠሎች", "ቅርጽ የሌለው፣ መራራ፣ ትንሽ ፍሬ በአንደኛው ጫፍ አረንጓዴ ሆኖ ሊቀር ይችላል", "የቅርንጫፍ መድረቅ፣ በመጨረሻም የዛፍ ሞት"], "cultural_control_header": "አያያዝ (ምንም ፈውስ የለም):", "cultural_control_list": ["ስርጭትን ለመከላከል ጥብቅ የኳራንቲን እርምጃዎች", "የእስያ የሎሚ ፕሲሊድ ተሸካሚ መቆጣጠር (የፀረ-ተባይ አተገባበር፣ ባዮሎጂያዊ ቁጥጥር)", "የበሽታውን ምንጭ ለመቀነስ የተበከሉ ዛፎችን ማስወገድ እና ማጥፋት", "የተረጋገጡ ከበሽታ ነጻ የሆኑ ዛፎችን መትከል"], "chemical_control_header": "የተሸካሚ ቁጥጥር:", "chemical_control_text": "በተጎዱ አካባቢዎች የፕሲሊድ ቁጥጥር ለማድረግ መደበኛ የፀረ-ተባይ አተገባበር ወሳኝ ነው። የአካባቢ የሎሚ አማካሪዎችን ያማክሩ እና ደንቦችን በጥብቅ ይከተሉ።", "further_info_link": ""},
        "ti": {"description": "ከቢድ ባክቴርያዊ ሕማም ኦም ሊባኖስ፡ ብኤስያዊት ፕሲሊድ ሊባኖስ ዝመሓላለፍ። ሕጂ ዘይፍወስ።", "symptoms_list": ["ምብጻሕ ቀጸልቲ፡ መብዛሕትኡ ግዜ ዘይስሩዕን ዘይማዕረን (ካብ ጉድለት መግቢ ዝተፈልየ)", "ንኣሽቱ፡ ቀጥ ዝበሉ ቀጸልቲ", "ቅርጺ ዘይብሉ፡ መሪር፡ ንእሽቶ ፍረ ኣብ ሓደ ወገን ቀጠልያ ኮይኑ ክተርፍ ዝኽእል", "ምድራቕ ጨናፍር፡ ኣብ መወዳእታ ሞት ኦም"], "cultural_control_header": "ምሕደራ (ፈውሲ የብሉን):", "cultural_control_list": ["ምስፍሕፋሕ ንምክልኻል ጽኑዕ ናይ ውሸባ ስጉምትታት", "ምቁጽጻር ናይ ኤስያዊት ፕሲሊድ ሊባኖስ (ኣጠቓቕማ ተባይ መከላኸሊ፡ ስነ-ህይወታዊ ምቁጽጻር)", "ምንጪ ሕማም ንምንካይ ዝተበከሉ ኣእዋም ምእላይን ምጥፋእን", "ካብ ሕማም ነጻ ዝኾኑ ኣእዋም ምትካል"], "chemical_control_header": "ምቁጽጻር ተሸካሚ ሕማም:", "chemical_control_text": "ኣብ ዝተጎድኡ ከባቢታት ንምቁጽጻር ፕሲሊድ፡ ስሩዕ ኣጠቓቕማ ተባይ መከላኸሊ ኣገዳሲ እዩ። ናይ ከባቢኹም ኣማኸርቲ ሊባኖስ ተወከሱን ደንብታት ብትኽክል ተኸተሉን።", "further_info_link": ""}
    },
    "Peach___Bacterial_spot": {
        "en": {"description": "Bacterial disease affecting peach leaves, fruit, and twigs.", "symptoms_list": ["Small, angular, water-soaked spots on leaves that turn purple, then brown, and may fall out ('shot-hole' appearance)", "Sunken, dark spots on fruit, sometimes with cracking", "Twig cankers in spring"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Plant resistant varieties", "Prune for good air circulation", "Avoid excessive nitrogen", "Maintain tree vigor", "Avoid working with trees when foliage is wet"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Copper-based bactericides applied during dormancy and early spring can help reduce inoculum. Some antibiotics may be used during the season. Consult local experts for timing and products.", "further_info_link": ""},
        "am": {"description": "የኮክ ቅጠሎችን፣ ፍራፍሬዎችን እና ቀንበጦችን የሚያጠቃ የባክቴሪያ በሽታ።", "symptoms_list": ["በቅጠሎች ላይ ትናንሽ፣ አንጉላር፣ ውሃ የዘፈዘፋቸው ነጠብጣቦች ወደ ወይን ጠጅ፣ ከዚያም ቡናማ የሚቀየሩ እና ሊወድቁ የሚችሉ ('የተተኮሰ ቀዳዳ' መልክ)", "በፍራፍሬ ላይ የሰመጡ፣ ጥቁር ነጠብጣቦች፣ አንዳንዴም መሰንጠቅ ያለባቸው", "በፀደይ ወቅት የቀንበጥ ካንከሮች"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ ዝርያዎችን ይትከሉ", "ለጥሩ የአየር ዝውውር መግረዝ", "ከመጠን በላይ ናይትሮጅንን ያስወግዱ", "የዛፍ ጥንካሬን ይጠብቁ", "ቅጠሎቹ እርጥብ ሲሆኑ ከዛፎች ጋር ከመሥራት ይቆጠቡ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "በእንቅልፍ እና በፀደይ መጀመሪያ ላይ የሚተገበሩ በመዳብ ላይ የተመሰረቱ ባክቴሪያ መድኃኒቶች የበሽታውን ምንጭ ለመቀነስ ይረዳሉ። አንዳንድ አንቲባዮቲኮች በወቅቱ ጥቅም ላይ ሊውሉ ይችላሉ። ለጊዜ አጠባበቅ እና ምርቶች የአካባቢ ባለሙያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ባክቴርያዊ ሕማም ንቀጽሊ፡ ፍረ፡ ጨናፍር ኮኽ ዝጎድእ።", "symptoms_list": ["ኣብ ቀጸልቲ ንኣሽቱ፡ ኩርናዓውያን፡ ማይ ዝሓዙ ነጠብጣባት ናብ ወይን ዝሕብሩ፡ ድሕሪኡ ናብ ቡናዊ ዝቕየሩን ክወድቑ ዝኽእሉን ('ከም ተተኲሱ ዝበደዐ' ዝመስል ትርኢት)", "ኣብ ፍረታት ዝጠሓሉ፡ ጸለምቲ ነጠብጣባት፡ ሓሓሊፉ ምስ ምንጣዕ", "ኣብ ጽድያ ናይ ጨናፍር ካንከራት"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ ዝርኣያት ተኸሉ", "ንጽቡቕ መዘዋወሪ ኣየር ምፍላጥ", "ልዑል ናይትሮጅን ኣወግዱ", "ሓይሊ ኦም ዓቅቡ", "ቆጽሊ ጥሉል ኣብ ዝኾነሉ እዋን ምስ ኣእዋም ካብ ምስራሕ ተቖጠቡ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ኣብ እዋን ዕረፍትን መጀመርታ ጽድያን ዝውዕሉ ኣብ ነሓስ ዝተመርኰሱ ባክቴርያ መከላኸሊ መድሃኒታት ንምንጪ ሕማም ንምንካይ ክሕግዙ ይኽእሉ። ገለ ኣንቲባዮቲካት ኣብ ወቕቲ ክጥቀሙ ይኽእሉ። ንግዜን ፍርያትን ምስ ናይ ከባቢኹም ክኢላታት ተማኸሩ።", "further_info_link": ""}
    },
    "Peach___healthy": {
        "en": {"description": "The peach plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Peach Care:", "cultural_control_list": ["Well-drained soil", "Full sun", "Proper pruning for fruit production and air circulation", "Thinning fruit", "Pest and disease monitoring (e.g., peach leaf curl, brown rot)"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Dormant sprays (e.g., copper or lime-sulfur) are important for preventing diseases like peach leaf curl. Follow local spray guides.", "further_info_link": ""},
        "am": {"description": "የኮክ ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የኮክ እንክብካቤ:", "cultural_control_list": ["ውሃ በደንብ የሚያሳልፍ አፈር", "ሙሉ ፀሐይ", "ለፍራፍሬ ምርት እና ለአየር ዝውውር ትክክለኛ መግረዝ", "ፍራፍሬን ማቃለል", "የተባይ እና የበሽታ ክትትል (ለምሳሌ፦ የኮክ ቅጠል መጠቅለል፣ ቡናማ መበስበስ)"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "የእንቅልፍ መርጫዎች (ለምሳሌ፦ መዳብ ወይም የኖራ-ሰልፈር) እንደ የኮክ ቅጠል መጠቅለል ያሉ በሽታዎችን ለመከላከል አስፈላጊ ናቸው። የአካባቢ የመርጨት መመሪያዎችን ይከተሉ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ ኮኽ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ኮኽ:", "cultural_control_list": ["ማይ ጽቡቕ ዘንጠብጥብ ሓመድ", "ሙሉእ ጸሓይ", "ንፍርያት ፍረን መዘዋወሪ ኣየርን ግቡእ ምፍላጥ", "ፍረ ምቅላል", "ምቁጽጻር ተሃሳስን ሕማምን (ከም ኣብነት፡ ምጭብጫብ ቀጽሊ ኮኽ፡ ቡናዊ ምብስባስ)"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ እዋን ዕረፍቲ ዝግበሩ መረጻሕቲ (ከም ኣብነት፡ ነሓስ ወይ ላይም-ሰልፈር) ከም ምጭብጫብ ቀጽሊ ኮኽ ዝኣመሰሉ ሕማማት ንምክልኻል ኣገደስቲ እዮም። ናይ ከባቢኹም መምርሒታት መረጻሕቲ ተኸተሉ።", "further_info_link": ""}
    },
    "Pepper,_bell___Bacterial_spot": {
        "en": {"description": "Bacterial disease causing spots on leaves and fruit of peppers.", "symptoms_list": ["Small, water-soaked spots on leaves that may turn dark brown to black, often with a yellow halo; leaf spots may fall out creating a shot-hole effect", "Raised, scab-like spots on fruit", "Stem lesions possible"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Use certified disease-free seed and transplants", "Plant resistant varieties", "Rotate crops (avoid planting peppers, tomatoes, eggplants in the same spot for 3-4 years)", "Improve air circulation", "Avoid working with plants when wet", "Remove and destroy infected plants/debris"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Copper-based bactericides can provide some protection if applied preventatively. Streptomycin (in some areas, with restrictions) may be used. Consult local experts.", "further_info_link": ""},
        "am": {"description": "በበርበሬ ቅጠሎች እና ፍራፍሬዎች ላይ ነጠብጣቦችን የሚያስከትል የባክቴሪያ በሽታ።", "symptoms_list": ["በቅጠሎች ላይ ትናንሽ፣ ውሃ የዘፈዘፋቸው ነጠብጣቦች ወደ ጥቁር ቡኒ እስከ ጥቁር ሊቀየሩ የሚችሉ፣ ብዙውን ጊዜ ቢጫ ቀለበት ያላቸው፤ የቅጠል ነጠብጣቦች ወድቀው የተተኮሰ ቀዳዳ ውጤት ሊፈጥሩ ይችላሉ", "በፍራፍሬ ላይ ከፍ ያሉ፣ እከክ መሰል ነጠብጣቦች", "የግንድ ቁስሎች ሊኖሩ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የተረጋገጡ ከበሽታ ነጻ የሆኑ ዘሮችን እና ችግኞችን ይጠቀሙ", "የሚቋቋሙ ዝርያዎችን ይትከሉ", "ሰብሎችን ያሽከርክሩ (በርበሬ፣ ቲማቲም፣ የእንቁላል ተክል በተመሳሳይ ቦታ ለ3-4 ዓመታት ከመትከል ይቆጠቡ)", "የአየር ዝውውርን ያሻሽሉ", "እርጥብ በሚሆኑበት ጊዜ ከእጽዋት ጋር ከመሥራት ይቆጠቡ", "የተበከሉ ተክሎችን/ፍርስራሾችን ያስወግዱ እና ያጥፉ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "በመዳብ ላይ የተመሰረቱ ባክቴሪያ መድኃኒቶች በመከላከል ከተተገበሩ የተወሰነ ጥበቃ ሊሰጡ ይችላሉ። ስትሬፕቶማይሲን (በአንዳንድ አካባቢዎች፣ ገደቦች ባሉበት) ጥቅም ላይ ሊውል ይችላል። የአካባቢ ባለሙያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ባክቴርያዊ ሕማም ኣብ ቀጽልን ፍረን በርበረ ነጠብጣባት ዘስዕብ።", "symptoms_list": ["ኣብ ቀጸልቲ ንኣሽቱ፡ ማይ ዝሓዙ ነጠብጣባት ናብ ጸሊም ቡናዊ ክሳብ ጸሊም ክቕየሩ ዝኽእሉ፡ መብዛሕትኡ ግዜ ምስ ብጫ ቀለቤት፤ ነጠብጣባት ቀጽሊ ወዲቖም ከም ተተኲሱ ዝበደዐ ውጽኢት ክፍጠር ይኽእል", "ኣብ ፍረታት ልዕል ዝበሉ፡ ከም ሕማም ስካብ ዝመስሉ ነጠብጣባት", "ቊስሊ ግንዲ ክኸውን ይኽእል"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርእን ተኽልታትን ተጠቐሙ", "ተጻወርቲ ዝርኣያት ተኸሉ", "ዘራእቲ ኣመሓይሹ (በርበረ፡ ቲማቲም፡ እንቋቝሖ ኣብ ተመሳሳሊ ቦታ ን3-4 ዓመታት ካብ ምትካል ተቖጠቡ)", "መዘዋወሪ ኣየር ኣመሓይሹ", "ተኽልታት ጥሉላት ኣብ ዝኾንሉ እዋን ካብ ምስራሕ ተቖጠቡ", "ዝተበከሉ ተኽልታት/ርስሓት ኣልዩን ኣጥፍኡን"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ኣብ ነሓስ ዝተመርኰሱ ባክቴርያ መከላኸሊ መድሃኒታት ከም መከላኸሊ እንተተጠቒሞም ገለ ዓይነት ዕቝባ ክህቡ ይኽእሉ። ስትሬፕቶማይሲን (ኣብ ገለ ከባቢታት፡ ምስ ገደባት) ክጥቀም ይኽእል። ናይ ከባቢኹም ክኢላታት ኣማኽሩ።", "further_info_link": ""}
    },
    "Pepper,_bell___healthy": {
        "en": {"description": "The bell pepper plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Pepper Care:", "cultural_control_list": ["Warm, sunny location", "Well-drained, fertile soil", "Consistent watering", "Support for plants (staking)", "Monitor for common pests (aphids, thrips) and diseases"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on good cultural practices. Use pesticides only if necessary based on scouting and expert advice.", "further_info_link": ""},
        "am": {"description": "የበርበሬ ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የበርበሬ እንክብካቤ:", "cultural_control_list": ["ሞቃት፣ ፀሐያማ ቦታ", "ውሃ በደንብ የሚያሳልፍ፣ ለም አፈር", "ወጥ የሆነ ውሃ ማጠጣት", "ለተክሎች ድጋፍ (መደገፍ)", "የተለመዱ ተባዮችን (አፊድ፣ ትሪፕስ) እና በሽታዎችን መከታተል"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በጥሩ የባህል ልምዶች ላይ ያተኩሩ። በቅኝት እና በባለሙያ ምክር ላይ በመመርኮዝ አስፈላጊ ከሆነ ብቻ ፀረ-ተባይ መድኃኒቶችን ይጠቀሙ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ በርበረ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን በርበረ:", "cultural_control_list": ["ሙቕ፡ ጸሓያዊ ቦታ", "ማይ ጽቡቕ ዘንጠብጥብ፡ ልሙዕ ሓመድ", "ቐጻሊ ምስታይ ማይ", "ንድጋፍ ተኽልታት (ምድጋፍ)", "ልሙዳት ተሃሳስ (ኣፊድ፡ ትሪፕስ)ን ሕማማትን ምቁጽጻር"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ጽቡቕ ባህላዊ ተግባራት ኣተኵሩ። ኣብ ምምርማርን ምኽሪ ክኢላታትን ተመርኲስኩም ኣድላዪ እንተኾይኑ ጥራይ ተባይ መከላኸሊ መድሃኒታት ተጠቐሙ።", "further_info_link": ""}
    },
    "Potato___Early_blight": {
        "en": {"description": "Fungal disease causing characteristic 'target-like' spots on potato leaves.", "symptoms_list": ["Dark brown to black spots on lower, older leaves, often with concentric rings ('target' or 'bulls-eye' pattern)", "Spots may be surrounded by a yellow halo", "Lesions can also occur on stems and tubers (dark, sunken, leathery spots)"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Use certified disease-free seed potatoes", "Plant resistant varieties", "Crop rotation (3-4 years away from potatoes/tomatoes)", "Adequate plant nutrition (especially nitrogen)", "Destroy infected crop debris and volunteer plants", "Proper hilling to protect tubers"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Foliar fungicides can be effective. Applications should begin when disease first appears or preventatively if conditions are favorable. Consult local potato advisors.", "further_info_link": ""},
        "am": {"description": "በድንች ቅጠሎች ላይ የ'ዒላማ' መሳይ ነጠብጣቦችን የሚያስከትል የፈንገስ በሽታ።", "symptoms_list": ["በታችኛው፣ በዕድሜ የገፉ ቅጠሎች ላይ ጥቁር ቡኒ እስከ ጥቁር ነጠብጣቦች፣ ብዙውን ጊዜ ከማዕከላዊ ቀለበቶች ጋር ('ዒላማ' ወይም 'የበሬ ዓይን' ስርዓተ-ጥለት)", "ነጠብጣቦች በቢጫ ቀለበት ሊከበቡ ይችላሉ", "ቁስሎች በግንዶች እና ሀረጎች ላይም ሊከሰቱ ይችላሉ (ጥቁር፣ የሰመጡ፣ የቆዳ መሳይ ነጠብጣቦች)"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የተረጋገጡ ከበሽታ ነጻ የሆኑ የዘር ድንች ይጠቀሙ", "የሚቋቋሙ ዝርያዎችን ይትከሉ", "የሰብል ማሽከርከር (ከድንች/ቲማቲም ለ3-4 ዓመታት የራቀ)", "በቂ የተክል ምግብ (በተለይ ናይትሮጅን)", "የተበከሉ የሰብል ፍርስራሾችን እና የበቀሉ ተክሎችን ማጥፋት", "ሀረጎችን ለመጠበቅ ትክክለኛ ኮትኮቶ ማድረግ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "የቅጠል ፈንገስ መድኃኒቶች ውጤታማ ሊሆኑ ይችላሉ። ትግበራዎች በሽታው ለመጀመሪያ ጊዜ ሲታይ ወይም ሁኔታዎች ምቹ ከሆኑ በመከላከል መጀመር አለባቸው። የአካባቢ የድንች አማካሪዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ኣብ ቀጽሊ ድንሽ ፍሉይ 'ዒላማ' ዝመስል ነጠብጣባት ዘስዕብ።", "symptoms_list": ["ኣብ ታሕተዎት፡ ዝኣረጉ ቀጸልቲ ጸሊም ቡናዊ ክሳብ ጸሊም ዝሕብሩ ነጠብጣባት፡ መብዛሕትኡ ግዜ ምስ ማእከላይ ቀለቤታት ('ዒላማ' ወይ 'ዒንዲ በዕራይ' ዝመስል ቅርጺ)", "ነጠብጣባት ብብጫ ቀለቤት ክኽበቡ ይኽእሉ", "ቊስልታት ኣብ ግንድን ሰራውርን'ውን ክርከቡ ይኽእሉ (ጸለምቲ፡ ዝጠሓሉ፡ ከም ሌጦ ዝመስሉ ነጠብጣባት)"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርኢ ድንሽ ተጠቐሙ", "ተጻወርቲ ዝርኣያት ተኸሉ", "ክቢ ዘራእቲ (ካብ ድንሽን ቲማቲምን ን3-4 ዓመታት ዝርሕቕ)", "እኹል መግቢ ተኽሊ (ብፍላይ ናይትሮጅን)", "ዝተበከለ ርስሓት ሰብልን ባዕሎም ዝበቘሉ ተኽልታትን ምጥፋእ", "ንሰራውር ንምክልኻል ግቡእ ምኹስኳስ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊታት ቀጽሊ ውጽኢታውያን ክኾኑ ይኽእሉ። ምጥቃም፡ ሕማም ንፈለማ ግዜ ኣብ ዝርኣየሉ እዋን ወይ ኩነታት ምቹእ እንተኾይኑ ከም መከላኸሊ ክጅምር ኣለዎ። ናይ ከባቢኹም ኣማኸርቲ ድንሽ ተወከሱ።", "further_info_link": ""}
    },
    "Potato___Late_blight": {
        "en": {"description": "Devastating disease (oomycete) affecting potatoes and tomatoes, especially in cool, moist weather.", "symptoms_list": ["Dark, water-soaked lesions on leaves/stems, often with white downy mold on undersides in humidity", "Lesions enlarge rapidly, killing foliage", "Tubers develop reddish-brown, dry rot that extends into the flesh"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Use certified disease-free seed potatoes", "Plant resistant varieties", "Destroy cull piles and volunteer plants", "Good air circulation, avoid dense planting", "Proper hilling", "Kill vines 2-3 weeks before harvest", "Harvest in dry weather, avoid wounding tubers"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "A strict preventative fungicide program is crucial in areas prone to late blight. Many effective products are available. Follow local advisories and consult experts.", "further_info_link": ""},
        "am": {"description": "ድንች እና ቲማቲሞችን የሚያጠቃ አውዳሚ በሽታ (ኦኦማይሴት) በተለይም በቀዝቃዛና እርጥበታማ የአየር ጠባይ።", "symptoms_list": ["በቅጠሎች/ግንዶች ላይ ጥቁር፣ ውሃ የዘፈዘፋቸው ቁስሎች፣ ብዙውን ጊዜ በእርጥበት ጊዜ በታችኛው ክፍል ላይ ነጭ የዱቄት ሻጋታ ያላቸው", "ቁስሎች በፍጥነት ይሰፋሉ፣ ቅጠሎችን ይገድላሉ", "ሀረጎች ቀይ-ቡኒ፣ ደረቅ መበስበስ ያዳብራሉ ይህም ወደ ሥጋው ይዘልቃል"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የተረጋገጡ ከበሽታ ነጻ የሆኑ የዘር ድንች ይጠቀሙ", "የሚቋቋሙ ዝርያዎችን ይትከሉ", "የተጣሉ ክምር እና የበቀሉ ተክሎችን ያጥፉ", "ጥሩ የአየር ዝውውር፣ ጥቅጥቅ ያለ መትከልን ያስወግዱ", "ትክክለኛ ኮትኮቶ ማድረግ", "ከመከሩ 2-3 ሳምንታት በፊት ግንዶችን መግደል", "በደረቅ የአየር ጠባይ መሰብሰብ፣ ሀረጎችን ከመጉዳት መቆጠብ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ዘግይቶ ለሚከሰት ብላይት በተጋለጡ አካባቢዎች ጥብቅ የመከላከያ ፈንገስ መድኃኒት ፕሮግራም ወሳኝ ነው። ብዙ ውጤታማ ምርቶች ይገኛሉ። የአካባቢ ምክሮችን ይከተሉ እና ባለሙያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ኣዕናዊ ሕማም (ኦኦማይሴት) ንድንሽን ቲማቲምን ዝጎድእ፡ ብፍላይ ኣብ ዝሑል፡ ጥሉል ኩነታት ኣየር።", "symptoms_list": ["ኣብ ቀጸልቲ/ግንዲ ጸለምቲ፡ ማይ ዝሓዙ ቊስልታት፡ መብዛሕትኡ ግዜ ኣብ ጥልቀት ኣብ ታሕተዋይ ገጽ ምስ ጻዕዳ ዱቄታዊ ሻጋታ", "ቊስልታት ብቕልጡፍ ይጋፍሑ፡ ንቆጽሊ ይቐትሉ", "ሰራውር ቀይሕ-ቡናዊ፡ ንቑጽ ምብስባስ የጥርዩ፡ እዚ ድማ ናብ ውሽጢ ስጋ ይዝርጋሕ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርኢ ድንሽ ተጠቐሙ", "ተጻወርቲ ዝርኣያት ተኸሉ", "ዝተደርበዩ ክምርን ባዕሎም ዝበቘሉ ተኽልታትን ኣጥፍኡ", "ጽቡቕ መዘዋወሪ ኣየር፡ ጽዑቕ ምትካል ኣወግዱ", "ግቡእ ምኹስኳስ", "ቅድሚ ቀውዒ ብ2-3 ሳምንታት ንተኽሊ ምቕታል", "ኣብ ንቑጽ ኩነታት ኣየር ምእራይ፡ ንሰራውር ካብ ምጕዳእ ምክልኻል"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ዳሕረዋይ ብላይት ኣብ ዝልመደሉ ከባቢታት ጽኑዕ መከላኸሊ ናይ ፈንገስ መከላኸሊ ፕሮግራም ኣገዳሲ እዩ። ብዙሓት ውጽኢታውያን ፍርያት ኣለዉ። ናይ ከባቢኹም ምኽርታት ተኸተሉን ምስ ክኢላታት ተማኸሩን።", "further_info_link": ""}
    },
    "Potato___healthy": {
        "en": {"description": "The potato plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Potato Care:", "cultural_control_list": ["Well-drained, loose soil", "Use certified seed potatoes", "Proper hilling", "Consistent moisture, especially during tuber formation", "Control weeds and pests (e.g., Colorado potato beetle)"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on good cultural practices and seed quality. Use pesticides based on scouting and local recommendations.", "further_info_link": ""},
        "am": {"description": "የድንች ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የድንች እንክብካቤ:", "cultural_control_list": ["ውሃ በደንብ የሚያሳልፍ፣ ልቅ አፈር", "የተረጋገጡ የዘር ድንች ይጠቀሙ", "ትክክለኛ ኮትኮቶ ማድረግ", "ወጥ የሆነ እርጥበት፣ በተለይም ሀረግ በሚፈጠርበት ጊዜ", "የአረም እና የተባይ መቆጣጠሪያ (ለምሳሌ፦ የኮሎራዶ ድንች ጥንዚዛ)"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በጥሩ የባህል ልምዶች እና በዘር ጥራት ላይ ያተኩሩ። በቅኝት እና በአካባቢያዊ ምክሮች ላይ በመመርኮዝ ፀረ-ተባይ መድኃኒቶችን ይጠቀሙ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ ድንሽ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ድንሽ:", "cultural_control_list": ["ማይ ጽቡቕ ዘንጠብጥብ፡ ልሑል ሓመድ", "ካብ ሕማም ነጻ ዝኾኑ ዘርኢ ድንሽ ተጠቐሙ", "ግቡእ ምኹስኳስ", "ቐጻሊ ጥልቀት፡ ብፍላይ ኣብ እዋን ምፍራይ ሰራውር", "ምቁጽጻር ኣእዋምን ተሃሳስን (ከም ኣብነት፡ ናይ ኮሎራዶ ድንሽ ጥንዚዛ)"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ጽቡቕ ባህላዊ ተግባራትን ጽሬት ዘርእን ኣተኵሩ። ኣብ ምምርማርን ናይ ከባቢ ምኽርታትን ተመርኲስኩም ተባይ መከላኸሊ መድሃኒታት ተጠቐሙ።", "further_info_link": ""}
    },
    "Raspberry___healthy": {
        "en": {"description": "The raspberry plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Raspberry Care:", "cultural_control_list": ["Well-drained soil, full sun", "Proper pruning of canes (differs for summer-bearing vs. everbearing)", "Trellising or support", "Weed control and mulching", "Monitor for common pests and diseases (e.g., cane borers, fruit rots)"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Good sanitation and pruning are key. Fungicides may be needed for fruit rots if conditions are wet during ripening. Consult local guides.", "further_info_link": ""},
        "am": {"description": "የራስበሪ ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የራስበሪ እንክብካቤ:", "cultural_control_list": ["ውሃ በደንብ የሚያሳልፍ አፈር፣ ሙሉ ፀሐይ", "ትክክለኛ የቀንበጦች መግረዝ (በበጋ-ለሚያፈሩ እና ሁልጊዜ-ለሚያፈሩ ይለያያል)", "መደገፊያ ወይም ድጋፍ", "የአረም መቆጣጠሪያ እና መሸፈኛ ማድረግ", "የተለመዱ ተባዮችን እና በሽታዎችን መከታተል (ለምሳሌ፦ የቀንበጥ ቦረሮች፣ የፍራፍሬ መበስበስ)"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "ጥሩ ንፅህና እና መግረዝ ቁልፍ ናቸው። በሚበስልበት ጊዜ ሁኔታዎች እርጥብ ከሆኑ ለፍራፍሬ መበስበስ ፈንገስ መድኃኒቶች ሊያስፈልጉ ይችላሉ። የአካባቢ መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ ራስበሪ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ራስበሪ:", "cultural_control_list": ["ማይ ጽቡቕ ዘንጠብጥብ ሓመድ፡ ሙሉእ ጸሓይ", "ግቡእ ምፍላጥ ጨናፍር (ንናይ ክረምቲ ፍረ ዝህቡን ንኩሉ ግዜ ፍረ ዝህቡን ይፈላለ)", "ምድጋፍ ወይ ደገፍ", "ምቁጽጻር ኣእዋምን ምጉስጓስን", "ልሙዳት ተሃሳስን ሕማማትን ምቁጽጻር (ከም ኣብነት፡ ናይ ጨናፍር ቦረራት፡ ምብስባስ ፍረ)"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ጽቡቕ ጽሬትን ምፍላጥን ቀንዲ እዮም። ኣብ እዋን ምብሳል ኩነታት ጥሉል እንተኾይኑ፡ ንምብስባስ ፍረ ፈንገስ መከላኸሊ መድሃኒታት ከድልዩ ይኽእሉ። ናይ ከባቢኹም መምርሒታት ተወከሱ።", "further_info_link": ""}
    },
    "Soybean___healthy": {
        "en": {"description": "The soybean plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Soybean Care (Agronomic Practices):", "cultural_control_list": ["Proper planting date and seeding rate", "Good soil fertility and pH", "Weed management", "Insect and nematode management (if applicable)", "Crop rotation"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on good agronomic practices and variety selection. Seed treatments can protect against early-season diseases. Foliar fungicides/insecticides used based on scouting and economic thresholds. Consult agricultural advisors.", "further_info_link": ""},
        "am": {"description": "የአኩሪ አተር ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የአኩሪ አተር እንክብካቤ (የግብርና ልምዶች):", "cultural_control_list": ["ትክክለኛ የመትከያ ቀን እና የዘር መጠን", "ጥሩ የአፈር ለምነት እና ፒኤች", "የአረም አያያዝ", "የነፍሳት እና የኒማቶድ አያያዝ (የሚመለከተው ከሆነ)", "የሰብል ማሽከርከር"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በጥሩ የግብርና ልምዶች እና በዘር ምርጫ ላይ ያተኩሩ። የዘር ህክምናዎች ከወቅቱ መጀመሪያ በሽታዎች ሊከላከሉ ይችላሉ። የቅጠል ፈንገስ/ፀረ-ተባይ መድኃኒቶች በቅኝት እና በኢኮኖሚያዊ ገደቦች ላይ በመመርኮዝ ያገለግላሉ። የግብርና አማካሪዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ ኣድሪ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ኣድሪ (ግብራዊ ኣገባባት ሕርሻ):", "cultural_control_list": ["ግቡእ ናይ ምዝራእ ግዜን መጠን ዘርእን", "ጽቡቕ ልምዓት ሓመድን ፒኤችን", "ምሕደራ ኣእዋም", "ምሕደራ ተሃሳስን ኒማቶድን (ዝምልከት እንተኾይኑ)", "ክቢ ዘራእቲ"], "cultural_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ጽቡቕ ኣገባባት ሕርሻን ምርጫ ዝርኣያትን ኣተኵሩ። ኣገባባት ሕክምና ዘርኢ ካብ ናይ መጀመርታ ወቕቲ ሕማማት ክከላኸሉ ይኽእሉ። ናይ ቀጽሊ ፈንገስ/ተባይ መከላኸሊ መድሃኒታት ኣብ ምምርማርን ቁጠባዊ ደረታትን ተመርኲስካ ይጥቀሙ። ምስ ናይ ሕርሻ ኣማኸርቲ ተማኸሩ።", "further_info_link": ""}
    },
    "Squash___Powdery_mildew": {
        "en": {"description": "Common fungal disease affecting squash, pumpkins, and other cucurbits, appearing as white powdery spots.", "symptoms_list": ["White, powdery spots on leaves (often upper surface first), stems, and petioles", "Spots can enlarge and cover entire leaf surface", "Infected leaves may turn yellow, brown, and die prematurely"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Plant resistant varieties", "Ensure good air circulation (proper spacing, avoid dense foliage)", "Plant in sunny locations", "Avoid overhead irrigation if possible, or water early in the day so foliage dries quickly", "Remove heavily infected leaves if practical"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides (e.g., sulfur, potassium bicarbonate, horticultural oils, systemic products) can be effective. Apply at first sign or preventatively. Rotate fungicide types to prevent resistance. Consult local guides.", "further_info_link": ""},
        "am": {"description": "ዱባ፣ ዱባ እና ሌሎች ኩኩሪቢቶችን የሚያጠቃ የተለመደ የፈንገስ በሽታ ሲሆን እንደ ነጭ የዱቄት ነጠብጣቦች ይታያል።", "symptoms_list": ["በቅጠሎች (ብዙውን ጊዜ የላይኛው ገጽ መጀመሪያ)፣ ግንዶች እና ፔትዮሎች ላይ ነጭ፣ የዱቄት ነጠብጣቦች", "ነጠብጣቦች ሊሰፉ እና ሙሉውን የቅጠል ገጽ ሊሸፍኑ ይችላሉ", "የተበከሉ ቅጠሎች ወደ ቢጫ፣ ቡናማነት ሊቀየሩ እና ያለጊዜው ሊሞቱ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ ዝርያዎችን ይትከሉ", "ጥሩ የአየር ዝውውርን ያረጋግጡ (ትክክለኛ ክፍተት፣ ጥቅጥቅ ያለ ቅጠልን ያስወግዱ)", "ፀሐያማ በሆኑ ቦታዎች ይትከሉ", "ከተቻለ ከላይ የሚደረግ መስኖን ያስወግዱ፣ ወይም ቅጠሎቹ በፍጥነት እንዲደርቁ በማለዳ ውሃ ያጠጡ", "ከተቻለ በጠና የተበከሉ ቅጠሎችን ያስወግዱ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ፈንገስ መድኃኒቶች (ለምሳሌ፦ ሰልፈር፣ ፖታሺየም ባይካርቦኔት፣ የጓሮ አትክልት ዘይቶች፣ ሥርዓታዊ ምርቶች) ውጤታማ ሊሆኑ ይችላሉ። በመጀመሪያ ምልክት ላይ ወይም በመከላከል ይተግብሩ። የመቋቋም አቅምን ለመከላከል የፈንገስ መድኃኒት ዓይነቶችን ያሽከርክሩ። የአካባቢ መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ልሙድ ፈንገሳዊ ሕማም ንዱባ፡ ፋኑስ፡ ካልኦት ኩኩርቢታትን ዝጎድእ፡ ከም ጻዕዳ ዱቄታዊ ነጠብጣባት ዝርአ።", "symptoms_list": ["ኣብ ቀጸልቲ (መብዛሕትኡ ግዜ ላዕለዋይ ገጽ መጀመርታ)፡ ግንዲ፡ ፔትዮላት ጻዕዳ፡ ዱቄታዊ ነጠብጣባት", "ነጠብጣባት ክጋፍሑን ንሙሉእ ገጽ ቀጽሊ ክሽፍኑን ይኽእሉ", "ዝተበከሉ ቀጸልቲ ናብ ብጫ፡ ቡናዊ ክቕየሩን ብዘይ ግዚኦም ክሞቱን ይኽእሉ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ ዝርኣያት ተኸሉ", "ጽቡቕ መዘዋወሪ ኣየር ኣረጋግጹ (ግቡእ ቦታ ምትካል፡ ጽዑቕ ቆጽሊ ኣወግዱ)", "ኣብ ጸሓያዊ ቦታታት ተኸሉ", "ዝከኣል እንተኾይኑ ካብ ላዕሊ ዝግበር መስኖ ኣወግዱ፡ ወይ ቆጽሊ ብቕልጡፍ ንኽደርቕ ንግሆ ኣስትዩ", "ዝከኣል እንተኾይኑ ብርቱዕ ዝተበከሉ ቀጸልቲ ኣልዩ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊ መድሃኒታት (ከም ኣብነት፡ ሰልፈር፡ ፖታሽየም ባይካርቦኔት፡ ናይ ኣታኽልቲ ዘይትታት፡ ስነ-ስርዓታዊ ፍርያት) ውጽኢታውያን ክኾኑ ይኽእሉ። ኣብ ቀዳማይ ምልክት ወይ ከም መከላኸሊ ተጠቐሙ። ንምክልኻል ተጻዋርነት፡ ዓይነታት ፈንገስ መከላኸሊ መድሃኒታት ኣመሓይሹ። ናይ ከባቢኹም መምርሒታት ተወከሱ።", "further_info_link": ""}
    },
    "Strawberry___Leaf_scorch": {
        "en": {"description": "Fungal disease causing irregular purplish blotches on strawberry leaves.", "symptoms_list": ["Small, irregular purplish spots on upper leaf surface", "Spots enlarge and may merge; centers turn brown, then gray", "Leaves may appear 'scorched' or 'blotchy'", "Similar lesions can occur on petioles, runners, and fruit stalks"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Plant resistant varieties", "Ensure good air circulation and sunlight penetration (renovate beds after harvest, control weeds)", "Remove and destroy infected leaves and plant debris after harvest", "Maintain adequate soil moisture and fertility to reduce plant stress"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicide applications may be needed, especially in wet seasons or on susceptible varieties. Start sprays early in spring. Consult local strawberry production guides.", "further_info_link": ""},
        "am": {"description": "በእንጆሪ ቅጠሎች ላይ መደበኛ ያልሆኑ ሐምራዊ ነጠብጣቦችን የሚያስከትል የፈንገስ በሽታ።", "symptoms_list": ["በላይኛው የቅጠል ገጽ ላይ ትናንሽ፣ መደበኛ ያልሆኑ ሐምራዊ ነጠብጣቦች", "ነጠብጣቦች ይሰፋሉ እና ሊዋሃዱ ይችላሉ፤ መካከላቸው ወደ ቡናማ፣ ከዚያም ግራጫነት ይቀየራል", "ቅጠሎች 'የተቃጠሉ' ወይም 'የተቦጫጨቁ' ሊመስሉ ይችላሉ", "ተመሳሳይ ቁስሎች በፔትዮሎች፣ ሯጮች እና የፍራፍሬ ግንዶች ላይ ሊከሰቱ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ ዝርያዎችን ይትከሉ", "ጥሩ የአየር ዝውውር እና የፀሐይ ብርሃን መግባቱን ያረጋግጡ (ከመከሩ በኋላ አልጋዎችን ያድሱ፣ አረሞችን ይቆጣጠሩ)", "ከመከሩ በኋላ የተበከሉ ቅጠሎችን እና የተክል ፍርስራሾችን ያስወግዱ እና ያጥፉ", "የተክል ጭንቀትን ለመቀነስ በቂ የአፈር እርጥበት እና ለምነት ይጠብቁ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "የፈንገስ መድኃኒት አተገባበር ሊያስፈልግ ይችላል፣ በተለይም በእርጥብ ወቅቶች ወይም ለበሽታ ተጋላጭ በሆኑ ዝርያዎች ላይ። በፀደይ መጀመሪያ ላይ መርጨት ይጀምሩ። የአካባቢ የእንጆሪ ምርት መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ፈንገሳዊ ሕማም ኣብ ቀጽሊ እንጆሪ ዘይስሩዕ ወይን ዝሕብሩ ነጠብጣባት ዘስዕብ።", "symptoms_list": ["ኣብ ላዕለዋይ ገጽ ቀጽሊ ንኣሽቱ፡ ዘይስሩዓት ወይን ዝሕብሩ ነጠብጣባት", "ነጠብጣባት ይጋፍሑን ክሓብሩን ይኽእሉ፤ ማእከሎም ናብ ቡናዊ፡ ድሕሪኡ ናብ ግራጫ ይቕየር", "ቀጸልቲ 'ከም ዝተቓጸሉ' ወይ 'ከም ዝተበርዓጹ' ክመስሉ ይኽእሉ", "ተመሳሳሊ ቊስልታት ኣብ ፔትዮላት፡ ሯነራት፡ ግንዲ ፍረታት ክርከቡ ይኽእሉ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ ዝርኣያት ተኸሉ", "ጽቡቕ መዘዋወሪ ኣየርን ምብጻሕ ብርሃን ጸሓይን ኣረጋግጹ (ድሕሪ ቀውዒ ንምንጣቓት ሓድሱ፡ ኣእዋም ምቁጽጻር)", "ድሕሪ ቀውዒ ዝተበከሉ ቀጸልትን ርስሓት ተኽልን ኣልዩን ኣጥፍኡን", "ጸቕጢ ተኽሊ ንምንካይ እኹል ጥልቀት ሓመድን ልምዓትን ዓቅቡ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ኣጠቓቕማ ፈንገስ መከላኸሊ መድሃኒት ከድሊ ይኽእል፡ ብፍላይ ኣብ ጥሉላት ወቕታት ወይ ኣብ ተቓለዕቲ ዝርኣያት። ኣብ መጀመርታ ጽድያ መረጻሕቲ ጀምሩ። ናይ ከባቢኹም ናይ እንጆሪ ምህርቲ መምርሒታት ተወከሱ።", "further_info_link": ""}
    },
    "Strawberry___healthy": {
        "en": {"description": "The strawberry plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Strawberry Care:", "cultural_control_list": ["Well-drained soil, full sun", "Proper planting system (matted row, plasticulture, etc.)", "Mulching to conserve moisture and suppress weeds", "Renovation of beds after harvest (for June-bearers)", "Pest and disease monitoring (e.g., fruit rots, spider mites)"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on cultural practices. Fungicides often needed for fruit rots, especially during wet weather at bloom and fruiting. Consult local guides.", "further_info_link": ""},
        "am": {"description": "የእንጆሪ ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የእንጆሪ እንክብካቤ:", "cultural_control_list": ["ውሃ በደንብ የሚያሳልፍ አፈር፣ ሙሉ ፀሐይ", "ትክክለኛ የመትከያ ስርዓት (የተነጠፈ ረድፍ፣ ፕላስቲካልቸር፣ ወዘተ)", "እርጥበትን ለመጠበቅ እና አረሞችን ለማፈን መሸፈኛ ማድረግ", "ከመከሩ በኋላ የአልጋዎች እድሳት (ለሰኔ-አፍቃሪዎች)", "የተባይ እና የበሽታ ክትትል (ለምሳሌ፦ የፍራፍሬ መበስበስ፣ የሸረሪት ሚይት)"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በባህላዊ ልምዶች ላይ ያተኩሩ። በተለይም በአበባ እና በፍራፍሬ ወቅት በእርጥብ የአየር ጠባይ ለፍራፍሬ መበስበስ ፈንገስ መድኃኒቶች ብዙውን ጊዜ ያስፈልጋሉ። የአካባቢ መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ እንጆሪ ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን እንጆሪ:", "cultural_control_list": ["ማይ ጽቡቕ ዘንጠብጥብ ሓመድ፡ ሙሉእ ጸሓይ", "ግቡእ ስርዓተ ምትካል (ዝተጸፈፈ መስርዕ፡ ፕላስቲካልቸር፡ ወዘተ)", "ንጥልቀት ንምዕቃብን ኣእዋም ንምዕፋንን ምጉስጓስ", "ድሕሪ ቀውዒ ንምንጣቓት ምሕዳስ (ንናይ ሰነ ፍረ ዝህቡ)", "ምቁጽጻር ተሃሳስን ሕማምን (ከም ኣብነት፡ ምብስባስ ፍረ፡ ሰራውር ሚይት)"], "cultural_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ባህላዊ ተግባራት ኣተኵሩ። ብፍላይ ኣብ እዋን ዕምባባን ምፍራይን ኣብ ጥሉል ኩነታት ኣየር፡ ንምብስባስ ፍረ ፈንገስ መከላኸሊ መድሃኒታት መብዛሕትኡ ግዜ የድልዩ። ናይ ከባቢኹም መምርሒታት ተወከሱ።", "further_info_link": ""}
    },
    "Tomato___Bacterial_spot": {
        "en": {"description": "Bacterial disease causing spots on tomato leaves, stems, and fruit.", "symptoms_list": ["Small, dark, water-soaked, angular spots on leaves, often with a yellow halo; centers may dry and fall out", "Raised, scab-like spots on fruit", "Stem lesions possible"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Use certified disease-free seed and transplants", "Plant resistant varieties", "Rotate crops (3-4 years away from tomato/pepper/eggplant)", "Improve air circulation", "Avoid working with plants when wet", "Remove and destroy infected plants/debris", "Stake or cage plants to keep foliage off the ground"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Copper-based bactericides can provide some protection if applied preventatively and regularly. Some products combine copper with mancozeb. Consult local experts.", "further_info_link": ""},
        "am": {"description": "በቲማቲም ቅጠሎች፣ ግንዶች እና ፍራፍሬዎች ላይ ነጠብጣቦችን የሚያስከትል የባክቴሪያ በሽታ።", "symptoms_list": ["በቅጠሎች ላይ ትናንሽ፣ ጥቁር፣ ውሃ የዘፈዘፋቸው፣ አንጉላር ነጠብጣቦች፣ ብዙውን ጊዜ ቢጫ ቀለበት ያላቸው፤ መካከላቸው ሊደርቅና ሊወድቅ ይችላል", "በፍራፍሬ ላይ ከፍ ያሉ፣ እከክ መሰል ነጠብጣቦች", "የግንድ ቁስሎች ሊኖሩ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የተረጋገጡ ከበሽታ ነጻ የሆኑ ዘሮችን እና ችግኞችን ይጠቀሙ", "የሚቋቋሙ ዝርያዎችን ይትከሉ", "ሰብሎችን ያሽከርክሩ (ከቲማቲም/በርበሬ/የእንቁላል ተክል ለ3-4 ዓመታት የራቀ)", "የአየር ዝውውርን ያሻሽሉ", "እርጥብ በሚሆኑበት ጊዜ ከእጽዋት ጋር ከመሥራት ይቆጠቡ", "የተበከሉ ተክሎችን/ፍርስራሾችን ያስወግዱ እና ያጥፉ", "ቅጠሎችን ከመሬት ላይ ለማራቅ ተክሎችን መደገፍ ወይም ማጠር"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "በመዳብ ላይ የተመሰረቱ ባክቴሪያ መድኃኒቶች በመከላከል እና በመደበኛነት ከተተገበሩ የተወሰነ ጥበቃ ሊሰጡ ይችላሉ። አንዳንድ ምርቶች መዳብን ከማንኮዜብ ጋር ያዋህዳሉ። የአካባቢ ባለሙያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ባክቴርያዊ ሕማም፡ ኣብ ቀጽሊ፡ ግንዲ፡ ፍረ ቲማቲም ነጠብጣባት ዘስዕብ።", "symptoms_list": ["ኣብ ቀጸልቲ ንኣሽቱ፡ ጸለምቲ፡ ማይ ዝሓዙ፡ ኩርናዓውያን ነጠብጣባት፡ መብዛሕትኡ ግዜ ምስ ብጫ ቀለቤት፤ ማእከሎም ክደርቕን ክወድቕን ይኽእል", "ኣብ ፍረታት ልዕል ዝበሉ፡ ከም ሕማም ስካብ ዝመስሉ ነጠብጣባት", "ቊስሊ ግንዲ ክኸውን ይኽእል"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርእን ተኽልታትን ተጠቐሙ", "ተጻወርቲ ዝርኣያት ተኸሉ", "ዘራእቲ ኣመሓይሹ (ካብ ቲማቲም/በርበረ/እንቋቝሖ ን3-4 ዓመታት ዝርሕቕ)", "መዘዋወሪ ኣየር ኣመሓይሹ", "ተኽልታት ጥሉላት ኣብ ዝኾንሉ እዋን ካብ ምስራሕ ተቖጠቡ", "ዝተበከሉ ተኽልታት/ርስሓት ኣልዩን ኣጥፍኡን", "ቆጽሊ ካብ ምድሪ ንምርሓቕ ንተኽልታት ደግፉ ወይ ዓጽዉዎም"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ኣብ ነሓስ ዝተመርኰሱ ባክቴርያ መከላኸሊ መድሃኒታት፡ ከም መከላኸሊን ብስሩዕን እንተተጠቒሞም፡ ገለ ዓይነት ዕቝባ ክህቡ ይኽእሉ። ገለ ፍርያት ነሓስ ምስ ማንኮዜብ የዋህድዎ። ናይ ከባቢኹም ክኢላታት ኣማኽሩ።", "further_info_link": ""}
    },
    "Tomato___Early_blight": {
        "en": {"description": "Fungal disease causing 'target-like' spots on tomato leaves and stem lesions.", "symptoms_list": ["Dark brown to black spots on lower leaves, often with concentric rings ('target' or 'bulls-eye' pattern)", "Spots enlarge, leaves may yellow and drop", "Dark, sunken lesions ('collar rot') can occur on stems near soil line", "Fruit can be infected, usually at the stem end (dark, leathery, sunken spots)"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Use certified disease-free seed/transplants", "Resistant varieties", "Crop rotation", "Good plant nutrition", "Sanitation: remove infected debris", "Stake/cage plants", "Mulch to reduce soil splash"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Foliar fungicides are effective. Apply preventatively or at first sign, especially if weather is favorable (warm, humid). Consult local tomato guides.", "further_info_link": ""},
        "am": {"description": "በቲማቲም ቅጠሎች ላይ 'የዒላማ' መሳይ ነጠብጣቦችን እና የግንድ ቁስሎችን የሚያስከትል የፈንገስ በሽታ።", "symptoms_list": ["በታችኛው ቅጠሎች ላይ ጥቁር ቡኒ እስከ ጥቁር ነጠብጣቦች፣ ብዙውን ጊዜ ከማዕከላዊ ቀለበቶች ጋር ('ዒላማ' ወይም 'የበሬ ዓይን' ስርዓተ-ጥለት)", "ነጠብጣቦች ይሰፋሉ፣ ቅጠሎች ወደ ቢጫነት ሊቀየሩ እና ሊረግፉ ይችላሉ", "ጥቁር፣ የሰመጡ ቁስሎች ('የአንገት መበስበስ') በአፈር መስመር አቅራቢያ ባሉ ግንዶች ላይ ሊከሰቱ ይችላሉ", "ፍራፍሬ ሊበከል ይችላል፣ ብዙውን ጊዜ በግንዱ ጫፍ (ጥቁር፣ የቆዳ መሳይ፣ የሰመጡ ነጠብጣቦች)"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የተረጋገጡ ከበሽታ ነጻ የሆኑ ዘሮችን/ችግኞችን ይጠቀሙ", "የሚቋቋሙ ዝርያዎችን ይትከሉ", "የሰብል ማሽከርከር", "ጥሩ የተክል ምግብ", "ንፅህና፦ የተበከሉ ፍርስራሾችን ያስወግዱ", "ተክሎችን መደገፍ/ማጠር", "የአፈር መፍሰስን ለመቀነስ መሸፈኛ ማድረግ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "የቅጠል ፈንገስ መድኃኒቶች ውጤታማ ናቸው። በመከላከል ወይም በመጀመሪያ ምልክት ላይ ይተግብሩ፣ በተለይም የአየር ሁኔታው ምቹ ከሆነ (ሞቃት፣ እርጥበታማ)። የአካባቢ የቲማቲም መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ኣብ ቀጽሊ ቲማቲም 'ዒላማ' ዝመስሉ ነጠብጣባትን ቊስሊ ግንድን ዘስዕብ።", "symptoms_list": ["ኣብ ታሕተዎት ቀጸልቲ ጸሊም ቡናዊ ክሳብ ጸሊም ዝሕብሩ ነጠብጣባት፡ መብዛሕትኡ ግዜ ምስ ማእከላይ ቀለቤታት ('ዒላማ' ወይ 'ዒንዲ በዕራይ' ዝመስል ቅርጺ)", "ነጠብጣባት ይጋፍሑ፡ ቀጸልቲ ናብ ብጫ ክቕየሩን ክረግፉን ይኽእሉ", "ጸለምቲ፡ ዝጠሓሉ ቊስልታት ('ምብስባስ ክሳድ') ኣብ ቀረባ መስመር ሓመድ ኣብ ግንዲ ክርከቡ ይኽእሉ", "ፍረ ክትለከፍ ይኽእል፡ መብዛሕትኡ ግዜ ኣብ ወሰን ግንዲ (ጸለምቲ፡ ከም ሌጦ ዝመስሉ፡ ዝጠሓሉ ነጠብጣባት)"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርኢ/ተኽልታት ተጠቐሙ", "ተጻወርቲ ዝርኣያት", "ክቢ ዘራእቲ", "ጽቡቕ መግቢ ተኽሊ", "ጽሬት፡ ዝተበከለ ርስሓት ምእላይ", "ንተኽልታት ደግፉ/ዓጽዉዎም", "ንትንታግ ሓመድ ንምንካይ ምጉስጓስ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊታት ቀጽሊ ውጽኢታውያን እዮም። ከም መከላኸሊ ወይ ኣብ ቀዳማይ ምልክት ተጠቐሙ፡ ብፍላይ ኩነታት ኣየር ምቹእ እንተኾይኑ (ሙቕ፡ ጥሉል)። ናይ ከባቢኹም ናይ ቲማቲም መምርሒታት ተወከሱ።", "further_info_link": ""}
    },
    "Tomato___Late_blight": {
        "en": {"description": "Devastating disease (oomycete) affecting tomatoes and potatoes, especially in cool, moist conditions.", "symptoms_list": ["Dark, water-soaked lesions on leaves/stems, often with white downy mold on undersides in humidity", "Lesions enlarge rapidly, killing foliage", "Fruit develops large, firm, greasy-looking, dark brown lesions"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Certified disease-free seed/transplants", "Resistant varieties", "Destroy cull/volunteer potato/tomato plants", "Good air circulation", "Stake/cage plants", "Avoid overhead irrigation", "Remove and destroy infected plants immediately"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "A strict preventative fungicide program is crucial. Follow local advisories and consult experts for products and timing.", "further_info_link": ""},
        "am": {"description": "ቲማቲሞችን እና ድንችን የሚያጠቃ አውዳሚ በሽታ (ኦኦማይሴት) በተለይም በቀዝቃዛና እርጥበታማ ሁኔታዎች።", "symptoms_list": ["በቅጠሎች/ግንዶች ላይ ጥቁር፣ ውሃ የዘፈዘፋቸው ቁስሎች፣ ብዙውን ጊዜ በእርጥበት ጊዜ በታችኛው ክፍል ላይ ነጭ የዱቄት ሻጋታ ያላቸው", "ቁስሎች በፍጥነት ይሰፋሉ፣ ቅጠሎችን ይገድላሉ", "ፍራፍሬ ትላልቅ፣ ጠንካራ፣ የቅባት መልክ ያላቸው፣ ጥቁር ቡኒ ቁስሎች ያዳብራል"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የተረጋገጡ ከበሽታ ነጻ የሆኑ ዘሮችን/ችግኞችን ይጠቀሙ", "የሚቋቋሙ ዝርያዎችን ይትከሉ", "የተጣሉ/የበቀሉ የድንች/ቲማቲም ተክሎችን ያጥፉ", "ጥሩ የአየር ዝውውር", "ተክሎችን መደገፍ/ማጠር", "ከላይ የሚደረግ መስኖን ያስወግዱ", "የተበከሉ ተክሎችን ወዲያውኑ ያስወግዱ እና ያጥፉ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ጥብቅ የመከላከያ ፈንገስ መድኃኒት ፕሮግራም ወሳኝ ነው። የአካባቢ ምክሮችን ይከተሉ እና ለምርቶች እና ጊዜ አጠባበቅ ባለሙያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ኣዕናዊ ሕማም (ኦኦማይሴት) ንቲማቲምን ድንሽን ዝጎድእ፡ ብፍላይ ኣብ ዝሑል፡ ጥሉል ኩነታት።", "symptoms_list": ["ኣብ ቀጸልቲ/ግንዲ ጸለምቲ፡ ማይ ዝሓዙ ቊስልታት፡ መብዛሕትኡ ግዜ ኣብ ጥልቀት ኣብ ታሕተዋይ ገጽ ምስ ጻዕዳ ዱቄታዊ ሻጋታ", "ቊስልታት ብቕልጡፍ ይጋፍሑ፡ ንቆጽሊ ይቐትሉ", "ፍረ ዓበይቲ፡ ጠጠው ዝበሉ፡ ከም ቅብኢ ዝመስሉ፡ ጸሊም ቡናዊ ቊስልታት የጥርዩ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርኢ/ተኽልታት", "ተጻወርቲ ዝርኣያት", "ዝተደርበዩ/ባዕሎም ዝበቘሉ ናይ ድንሽ/ቲማቲም ተኽልታት ምጥፋእ", "ጽቡቕ መዘዋወሪ ኣየር", "ንተኽልታት ደግፉ/ዓጽዉዎም", "ካብ ላዕሊ ዝግበር መስኖ ምውጋድ", "ዝተበከሉ ተኽልታት ብቕጽበት ኣልዩን ኣጥፍኡን"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ጽኑዕ መከላኸሊ ናይ ፈንገስ መከላኸሊ ፕሮግራም ኣገዳሲ እዩ። ናይ ከባቢኹም ምኽርታት ተኸተሉን ንፍርያትን ግዜን ምስ ክኢላታት ተማኸሩን።", "further_info_link": ""}
    },
    "Tomato___Leaf_Mold": {
        "en": {"description": "Fungal disease common in greenhouse tomatoes or high humidity, causing yellow spots on upper leaf surfaces and olive-green mold on lower surfaces.", "symptoms_list": ["Pale green or yellowish spots on upper leaf surfaces", "Olive-green to brownish, velvety mold patches on the corresponding lower leaf surfaces", "Leaves may curl, wither, and drop prematurely"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Resistant varieties (many exist, but new fungal races can overcome resistance)", "Ensure excellent air circulation (spacing, pruning, ventilation in greenhouses)", "Reduce humidity (avoid overhead watering, water early, vent greenhouses)", "Remove and destroy infected leaves/plants"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides can be used, especially protectant fungicides. Good coverage is important. Consult local guides, especially for greenhouse use.", "further_info_link": ""},
        "am": {"description": "በግሪንሀውስ ቲማቲሞች ወይም በከፍተኛ እርጥበት የተለመደ የፈንገስ በሽታ ሲሆን በላይኛው የቅጠል ገጽ ላይ ቢጫ ነጠብጣቦችን እና በታችኛው ገጽ ላይ የወይራ-አረንጓዴ ሻጋታ ያስከትላል።", "symptoms_list": ["በላይኛው የቅጠል ገጽ ላይ ፈዛዛ አረንጓዴ ወይም ቢጫማ ነጠብጣቦች", "በተዛማጁ የታችኛው የቅጠል ገጽ ላይ የወይራ-አረንጓዴ እስከ ቡናማ፣ የቬልቬት ሻጋታ ምልክቶች", "ቅጠሎች ሊጠቀለሉ፣ ሊደርቁ እና ያለጊዜው ሊረግፉ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["የሚቋቋሙ ዝርያዎች (ብዙ አሉ፣ ነገር ግን አዲስ የፈንገስ ዝርያዎች የመቋቋም አቅምን ሊያሸንፉ ይችላሉ)", "እጅግ በጣም ጥሩ የአየር ዝውውርን ያረጋግጡ (ክፍተት፣ መግረዝ፣ በግሪንሀውስ ውስጥ አየር ማናፈሻ)", "እርጥበትን ይቀንሱ (ከላይ ውሃ ማጠጣትን ያስወግዱ፣ በማለዳ ውሃ ያጠጡ፣ ግሪንሀውስን ያናፍሱ)", "የተበከሉ ቅጠሎችን/ተክሎችን ያስወግዱ እና ያጥፉ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ፈንገስ መድኃኒቶች ጥቅም ላይ ሊውሉ ይችላሉ፣ በተለይም የመከላከያ ፈንገስ መድኃኒቶች። ጥሩ ሽፋን አስፈላጊ ነው። የአካባቢ መመሪያዎችን ያማክሩ፣ በተለይም ለግሪንሀውስ አጠቃቀም።", "further_info_link": ""},
        "ti": {"description": "ፈንገሳዊ ሕማም ኣብ ናይ ግሪንሃውስ ቲማቲም ወይ ልዑል ጥልቀት ልሙድ ዝኾነ፡ ኣብ ላዕለዋይ ገጽ ቀጽሊ ብጫ ነጠብጣባትን ኣብ ታሕተዋይ ገጽ ወይራ-ቀጠልያ ሻጋታን ዘስዕብ።", "symptoms_list": ["ኣብ ላዕለዋይ ገጽ ቀጽሊ ፈኲስ ቀጠልያ ወይ ብጫ ዝሕብሩ ነጠብጣባት", "ኣብ ተዛማዲ ታሕተዋይ ገጽ ቀጽሊ ወይራ-ቀጠልያ ክሳብ ቡናዊ፡ ከም ቬልቬት ዝበለ ናይ ሻጋታ ምልክታት", "ቀጸልቲ ክጥምጠሙ፡ ክደርቑ፡ ብዘይ ግዚኦም ክረግፉ ይኽእሉ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ተጻወርቲ ዝርኣያት (ብዙሓት ኣለዉ፡ ግን ሓደስቲ ዓሌታት ፈንገስ ንተጻዋርነት ክስዕርዎ ይኽእሉ)", "ብሉጽ መዘዋወሪ ኣየር ኣረጋግጹ (ቦታ ምትካል፡ ምፍላጥ፡ ኣየር ምውጻእ ኣብ ግሪንሃውሳት)", "ጥልቀት ቀንስ (ካብ ላዕሊ ምስታይ ማይ ኣወግዱ፡ ንግሆ ኣስትዩ፡ ግሪንሃውሳት ኣየር ኣውጽኡ)", "ዝተበከሉ ቀጸልቲ/ተኽልታት ኣልዩን ኣጥፍኡን"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ፈንገስ መከላኸሊ መድሃኒታት ክጥቀሙ ይኽእሉ፡ ብፍላይ መከላኸሊ ፈንገስ መከላኸሊ መድሃኒታት። ጽቡቕ ምሽፋን ኣገዳሲ እዩ። ናይ ከባቢኹም መምርሒታት ተወከሱ፡ ብፍላይ ንኣጠቓቕማ ግሪንሃውስ።", "further_info_link": ""}
    },
    "Tomato___Septoria_leaf_spot": {
        "en": {"description": "Common fungal disease of tomatoes causing numerous small spots with dark borders and light centers on leaves.", "symptoms_list": ["Many small (1/16-1/8 inch), circular spots on older, lower leaves first", "Spots have dark brown margins and tan to gray centers; tiny black fungal fruiting bodies (pycnidia) may be visible in centers", "Spots can merge, leaves yellow, wither, and drop"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Use disease-free seed/transplants", "Crop rotation (3-4 years)", "Sanitation: remove infected debris, till deeply", "Improve air circulation (staking, pruning)", "Mulch to reduce soil splash", "Avoid overhead irrigation"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides can be effective if applied when disease first appears and repeated as needed. Consult local tomato guides for product recommendations.", "further_info_link": ""},
        "am": {"description": "በቲማቲም ቅጠሎች ላይ ጥቁር ድንበር እና ቀላል ማዕከል ያላቸው በርካታ ትናንሽ ነጠብጣቦችን የሚያስከትል የተለመደ የፈንገስ በሽታ።", "symptoms_list": ["በመጀመሪያ በዕድሜ የገፉ፣ በታችኛው ቅጠሎች ላይ ብዙ ትናንሽ (1/16-1/8 ኢንች)፣ ክብ ነጠብጣቦች", "ነጠብጣቦች ጥቁር ቡናማ ድንበር እና ፈዛዛ ቡኒ እስከ ግራጫ ማዕከሎች አሏቸው፤ በማዕከሎች ውስጥ ትናንሽ ጥቁር የፈንገስ ፍሬያማ አካላት (ፒክኒዲያ) ሊታዩ ይችላሉ", "ነጠብጣቦች ሊዋሃዱ ይችላሉ፣ ቅጠሎች ወደ ቢጫነት ይቀየራሉ፣ ይደርቃሉ እና ይረግፋሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["ከበሽታ ነጻ የሆኑ ዘሮችን/ችግኞችን ይጠቀሙ", "የሰብል ማሽከርከር (3-4 ዓመታት)", "ንፅህና፦ የተበከሉ ፍርስራሾችን ያስወግዱ፣ በጥልቀት ያርሱ", "የአየር ዝውውርን ያሻሽሉ (መደገፍ፣ መግረዝ)", "የአፈር መፍሰስን ለመቀነስ መሸፈኛ ማድረግ", "ከላይ የሚደረግ መስኖን ያስወግዱ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "በሽታው ለመጀመሪያ ጊዜ ሲታይ ከተተገበሩ እና እንደአስፈላጊነቱ ከተደጋገሙ ፈንገስ መድኃኒቶች ውጤታማ ሊሆኑ ይችላሉ። ለምርት ምክሮች የአካባቢ የቲማቲም መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ልሙድ ፈንገሳዊ ሕማም ቲማቲም፡ ኣብ ቀጸልቲ ብዙሓት ንኣሽቱ ነጠብጣባት ምስ ጸሊም ደረትን ፈኲስ ማእከልን ዘስዕብ።", "symptoms_list": ["ኣብ ዝኣረጉ፡ ታሕተዎት ቀጸልቲ መጀመርታ ብዙሓት ንኣሽቱ (1/16-1/8 ኢንች)፡ ክብ ነጠብጣባት", "ነጠብጣባት ጸሊም ቡናዊ ወሰንን ሃመዳዊ ክሳብ ግራጫ ማእከልን ኣለዎም፤ ኣብ ማእከላት ንኣሽቱ ጸለምቲ ፈንገሳዊ ኣካላት ፍረ (ፒክኒድያ) ክርኣዩ ይኽእሉ", "ነጠብጣባት ክሓብሩ ይኽእሉ፡ ቀጸልቲ ናብ ብጫ ይቕየሩ፡ ይደርቑ፡ ይረግፉ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርኢ/ተኽልታት ተጠቐሙ", "ክቢ ዘራእቲ (3-4 ዓመታት)", "ጽሬት፡ ዝተበከለ ርስሓት ምእላይ፡ ዓሚቝ ምሕራስ", "መዘዋወሪ ኣየር ምምሕያሽ (ምድጋፍ፡ ምፍላጥ)", "ንትንታግ ሓመድ ንምንካይ ምጉስጓስ", "ካብ ላዕሊ ዝግበር መስኖ ምውጋድ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ሕማም ንፈለማ ግዜ ኣብ ዝርኣየሉ እዋን እንተተጠቒሞምን ከም ኣድላይነቱ እንተተደጊሞምን ፈንገስ መከላኸሊ መድሃኒታት ውጽኢታውያን ክኾኑ ይኽእሉ። ንምኽሪ ፍርያት ናይ ከባቢኹም ናይ ቲማቲም መምርሒታት ተወከሱ።", "further_info_link": ""}
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "en": {"description": "Tiny arachnid pests (not insects) that feed on tomato plant sap, causing stippling and webbing.", "symptoms_list": ["Fine yellow or white stippling (tiny dots) on upper leaf surfaces", "Leaves may appear bronzed or bleached", "Fine webbing on undersides of leaves or between leaves/stems in heavy infestations", "Tiny moving specks (mites) visible with a hand lens, especially on leaf undersides", "Premature leaf drop"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Monitor plants regularly, especially undersides of leaves", "Increase humidity (mites thrive in dry conditions)", "Strong water sprays to dislodge mites", "Remove heavily infested leaves/plants", "Encourage beneficial insects (predatory mites, ladybugs)", "Control weeds that can harbor mites"], "chemical_control_header": "Chemical Control (Miticides):", "chemical_control_text": "Insecticidal soaps or horticultural oils can be effective if coverage is good. Specific miticides may be needed for severe infestations. Rotate miticide types to prevent resistance. Consult local experts.", "further_info_link": ""},
        "am": {"description": "የቲማቲም ተክል ጭማቂን የሚመገቡ ትናንሽ የአራክኒድ ተባዮች (ነፍሳት አይደሉም) ሲሆኑ ነጠብጣቦችን እና ድርን ያስከትላሉ።", "symptoms_list": ["በላይኛው የቅጠል ገጽ ላይ ስስ ቢጫ ወይም ነጭ ነጠብጣቦች (ትናንሽ ነጥቦች)", "ቅጠሎች የነሐስ ወይም የነጣ መልክ ሊኖራቸው ይችላል", "በከባድ ወረርሽኝ ወቅት በቅጠሎች የታችኛው ክፍል ወይም በቅጠሎች/ግንዶች መካከል ስስ ድር", "በእጅ መነፅር የሚታዩ ትናንሽ ተንቀሳቃሽ ነጠብጣቦች (ሚይት)፣ በተለይም በቅጠሎች የታችኛው ክፍል ላይ", "ያለጊዜው የቅጠል መውደቅ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["ተክሎችን በመደበኛነት ይቆጣጠሩ፣ በተለይም የቅጠሎችን የታችኛው ክፍል", "እርጥበትን ይጨምሩ (ሚይት በደረቅ ሁኔታዎች ይበቅላል)", "ሚይትን ለማስወገድ ጠንካራ የውሃ መርጫዎች", "በጠና የተበከሉ ቅጠሎችን/ተክሎችን ያስወግዱ", "ጠቃሚ ነፍሳትን ያበረታቱ (አዳኝ ሚይት፣ ጥንዚዛዎች)", "ሚይትን ሊያስተናግዱ የሚችሉ አረሞችን ይቆጣጠሩ"], "chemical_control_header": "የኬሚካል ቁጥጥር (ሚቲሳይዶች):", "chemical_control_text": "ሽፋኑ ጥሩ ከሆነ የፀረ-ተባይ ሳሙናዎች ወይም የጓሮ አትክልት ዘይቶች ውጤታማ ሊሆኑ ይችላሉ። ለከባድ ወረርሽኝ የተወሰኑ ሚቲሳይዶች ሊያስፈልጉ ይችላሉ። የመቋቋም አቅምን ለመከላከል የሚቲሳይድ ዓይነቶችን ያሽከርክሩ። የአካባቢ ባለሙያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ንኣሽቱ ኣራክኒድ ተሃሳስ (ሸኻኺት ኣይኮኑን) ንፈሳሲ ተኽሊ ቲማቲም ዝምገቡ፡ ነጠብጣባትን መርበብን ዘስዕቡ።", "symptoms_list": ["ኣብ ላዕለዋይ ገጽ ቀጽሊ ቀጢን ብጫ ወይ ጻዕዳ ነጠብጣባት (ንኣሽቱ ነጥብታት)", "ቀጸልቲ ከም ነሓስ ዝበለ ወይ ዝነጽሀ ትርኢት ክህልዎም ይኽእል", "ኣብ ከቢድ ወራራት ኣብ ታሕተዋይ ገጽ ቀጽሊ ወይ ኣብ መንጎ ቀጸልትን ግንድን ቀጢን መርበብ", "ብናይ ኢድ መነጽር ዝርኣዩ ንኣሽቱ ተንቀሳቐስቲ ነጠብጣባት (ሚይት)፡ ብፍላይ ኣብ ታሕተዋይ ገጽ ቀጽሊ", "ብዘይ ግዚኡ ምርጋፍ ቀጽሊ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ንተኽልታት ብስሩዕ መርምሩ፡ ብፍላይ ንታሕተዋይ ገጽ ቀጽሊ", "ጥልቀት ወስኹ (ሚይት ኣብ ንቑጽ ኩነታት ይዓቢ)", "ንሚይት ንምውጋድ ሓያል መረጻሕቲ ማይ", "ብርቱዕ ዝተበከሉ ቀጸልቲ/ተኽልታት ኣልዩ", "ጠቐምቲ ሸኻኺት ኣተባብዑ (ኣሃዲ ሚይት፡ ጥንዚዛ)", "ንሚይት ከዕቁቡ ዝኽእሉ ኣእዋም ምቁጽጻር"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር (ሚቲሳይዳት):", "chemical_control_text": "ምሽፋን ጽቡቕ እንተኾይኑ፡ ተባይ መከላኸሊ ሳሙና ወይ ናይ ኣታኽልቲ ዘይትታት ውጽኢታውያን ክኾኑ ይኽእሉ። ንኸቢድ ወራራት ፍሉያት ሚቲሳይዳት ከድልዩ ይኽእሉ። ንምክልኻል ተጻዋርነት፡ ዓይነታት ሚቲሳይዳት ኣመሓይሹ። ናይ ከባቢኹም ክኢላታት ኣማኽሩ።", "further_info_link": ""}
    },
    "Tomato___Target_Spot": {
        "en": {"description": "Fungal disease causing distinctive target-like spots on tomato leaves, stems, and fruit.", "symptoms_list": ["Small, dark spots on leaves that enlarge and develop concentric rings ('target' pattern), often with a dark center", "Lesions can also appear on stems and petioles", "Fruit lesions are typically sunken, dark, and may also show concentric rings"], "cultural_control_header": "Cultural Control & Prevention:", "cultural_control_list": ["Use disease-free seed/transplants", "Crop rotation", "Sanitation: remove and destroy infected plant debris", "Improve air circulation", "Avoid overhead irrigation", "Stake plants to keep foliage off ground"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "Fungicides used for other tomato foliar diseases (like early blight) often provide control. Apply preventatively or at first sign. Consult local tomato guides.", "further_info_link": ""},
        "am": {"description": "በቲማቲም ቅጠሎች፣ ግንዶች እና ፍራፍሬዎች ላይ የ'ዒላማ' መሳይ ነጠብጣቦችን የሚያስከትል የፈንገስ በሽታ።", "symptoms_list": ["በቅጠሎች ላይ ትናንሽ፣ ጥቁር ነጠብጣቦች እየሰፉ የሚሄዱ እና የማዕከላዊ ቀለበቶችን የሚያዳብሩ ('ዒላማ' ስርዓተ-ጥለት)፣ ብዙውን ጊዜ ጥቁር ማዕከል ያላቸው", "ቁስሎች በግንዶች እና በፔትዮሎች ላይም ሊታዩ ይችላሉ", "የፍራፍሬ ቁስሎች በአብዛኛው የሰመጡ፣ ጥቁር እና የማዕከላዊ ቀለበቶችን ሊያሳዩ ይችላሉ"], "cultural_control_header": "የባህል ቁጥጥር እና መከላከል:", "cultural_control_list": ["ከበሽታ ነጻ የሆኑ ዘሮችን/ችግኞችን ይጠቀሙ", "የሰብል ማሽከርከር", "ንፅህና፦ የተበከሉ የተክል ፍርስራሾችን ያስወግዱ እና ያጥፉ", "የአየር ዝውውርን ያሻሽሉ", "ከላይ የሚደረግ መስኖን ያስወግዱ", "ቅጠሎችን ከመሬት ላይ ለማራቅ ተክሎችን መደገፍ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ለሌሎች የቲማቲም ቅጠል በሽታዎች (እንደ ቀደምት ብላይት) የሚያገለግሉ ፈንገስ መድኃኒቶች ብዙውን ጊዜ ቁጥጥር ይሰጣሉ። በመከላከል ወይም በመጀመሪያ ምልክት ላይ ይተግብሩ። የአካባቢ የቲማቲም መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ፈንገሳዊ ሕማም፡ ኣብ ቀጽሊ፡ ግንዲ፡ ፍረ ቲማቲም ፍሉይ 'ዒላማ' ዝመስሉ ነጠብጣባት ዘስዕብ።", "symptoms_list": ["ኣብ ቀጸልቲ ንኣሽቱ፡ ጸለምቲ ነጠብጣባት እናገፍሑ ዝኸዱን ማእከላይ ቀለቤታት ዘጥርዩን ('ዒላማ' ዝመስል ቅርጺ)፡ መብዛሕትኡ ግዜ ምስ ጸሊም ማእከል", "ቊስልታት ኣብ ግንዲን ፔትዮላትን'ውን ክርኣዩ ይኽእሉ", "ቊስልታት ፍረ መብዛሕትኡ ግዜ ዝጠሓሉ፡ ጸለምቲ፡ ማእከላይ ቀለቤታት'ውን ከርእዩ ይኽእሉ"], "cultural_control_header": "ባህላዊ ምቁጽጻርን ምክልኻልን:", "cultural_control_list": ["ካብ ሕማም ነጻ ዝኾኑ ዘርኢ/ተኽልታት ተጠቐሙ", "ክቢ ዘራእቲ", "ጽሬት፡ ዝተበከለ ርስሓት ተኽሊ ኣልዩን ኣጥፍኡን", "መዘዋወሪ ኣየር ኣመሓይሹ", "ካብ ላዕሊ ዝግበር መስኖ ምውጋድ", "ቆጽሊ ካብ ምድሪ ንምርሓቕ ንተኽልታት ደግፉ"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ንኻልኦት ናይ ቲማቲም ቀጽሊ ሕማማት (ከም ቀዳማይ ብላይት) ዝውዕሉ ፈንገስ መከላኸሊ መድሃኒታት መብዛሕትኡ ግዜ ምቁጽጻር ይህቡ። ከም መከላኸሊ ወይ ኣብ ቀዳማይ ምልክት ተጠቐሙ። ናይ ከባቢኹም ናይ ቲማቲም መምርሒታት ተወከሱ።", "further_info_link": ""}
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "en": {"description": "A serious viral disease of tomatoes transmitted by whiteflies, causing stunting and leaf curling.", "symptoms_list": ["Upward curling and yellowing of leaf margins", "Stunted plant growth, reduced leaf size", "Flowers may drop, leading to poor fruit set", "Veins may appear purplish on leaf undersides"], "cultural_control_header": "Management (No Cure for Infected Plants):", "cultural_control_list": ["Control whitefly populations (primary vector). This is key.", "Use virus-resistant/tolerant tomato varieties", "Plant certified disease-free transplants", "Remove and destroy infected plants immediately to reduce virus spread", "Use reflective mulches to deter whiteflies", "Sanitation: control weeds that can host whiteflies or the virus"], "chemical_control_header": "Whitefly (Vector) Control:", "chemical_control_text": "Insecticides targeting whiteflies are essential for management. Systemic insecticides applied at planting or foliar sprays may be needed. Rotate insecticide modes of action. Consult local experts for whitefly management strategies.", "further_info_link": ""},
        "am": {"description": "በነጭ ዝንቦች የሚተላለፍ ከባድ የቲማቲም የቫይረስ በሽታ ሲሆን እድገትን ማቆም እና የቅጠል መጠቅለልን ያስከትላል።", "symptoms_list": ["የቅጠል ዳርቻዎች ወደ ላይ መጠቅለል እና ቢጫ መሆን", "የተቀነሰ የተክል እድገት፣ የተቀነሰ የቅጠል መጠን", "አበቦች ሊረግፉ ይችላሉ፣ ይህም ደካማ የፍራፍሬ አቀማመጥን ያስከትላል", "በቅጠሎች የታችኛው ክፍል ላይ የደም ሥሮች ሐምራዊ ሊመስሉ ይችላሉ"], "cultural_control_header": "አያያዝ (ለተበከሉ ተክሎች ምንም ፈውስ የለም):", "cultural_control_list": ["የነጭ ዝንብ ብዛትን መቆጣጠር (ዋና ተሸካሚ)። ይህ ቁልፍ ነው።", "ቫይረስን የሚቋቋሙ/የሚችሉ የቲማቲም ዝርያዎችን ይጠቀሙ", "የተረጋገጡ ከበሽታ ነጻ የሆኑ ችግኞችን ይትከሉ", "የቫይረስ ስርጭትን ለመቀነስ የተበከሉ ተክሎችን ወዲያውኑ ያስወግዱ እና ያጥፉ", "ነጭ ዝንቦችን ለማስወገድ አንጸባራቂ መሸፈኛዎችን ይጠቀሙ", "ንፅህና፦ ነጭ ዝንቦችን ወይም ቫይረሱን ሊያስተናግዱ የሚችሉ አረሞችን ይቆጣጠሩ"], "chemical_control_header": "የነጭ ዝንብ (ተሸካሚ) ቁጥጥር:", "chemical_control_text": "ነጭ ዝንቦችን ዒላማ ያደረጉ ፀረ-ተባይ መድኃኒቶች ለአያያዝ አስፈላጊ ናቸው። በሚተክሉበት ጊዜ የሚተገበሩ ሥርዓታዊ ፀረ-ተባይ መድኃኒቶች ወይም የቅጠል መርጫዎች ሊያስፈልጉ ይችላሉ። የፀረ-ተባይ የአሠራር ዘዴዎችን ያሽከርክሩ። ለነጭ ዝንብ አያያዝ ስልቶች የአካባቢ ባለሙያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "ከቢድ ቫይራዊ ሕማም ቲማቲም ብጻዕዳ ዝንቢ ዝመሓላለፍ፡ ምድንጓይ ዕቤትን ምጭብጫብ ቀጽልን ዘስዕብ።", "symptoms_list": ["ናብ ላዕሊ ምጭብጫብን ምብጻሕን ወሰን ቀጽሊ", "ዝተደንጎየ ዕቤት ተኽሊ፡ ዝነከየ ዓቐን ቀጽሊ", "ዕምባባታት ክረግፉ ይኽእሉ፡ እዚ ድማ ናብ ድኹም ምፍራይ ዘምርሕ", "ኣብ ታሕተዋይ ገጽ ቀጽሊ ሰራውር ወይን ዝሕብሩ ክመስሉ ይኽእሉ"], "cultural_control_header": "ምሕደራ (ንዝተበከሉ ተኽልታት ፈውሲ የብሎምን):", "cultural_control_list": ["ብዝሒ ጻዕዳ ዝንቢ ምቁጽጻር (ቀንዲ ተሸካሚ)። እዚ ቀንዲ ነገር እዩ።", "ንቫይረስ ዝጻወሩ/ዝኽእሉ ዝርኣያት ቲማቲም ተጠቐሙ", "ካብ ሕማም ነጻ ዝኾኑ ተኽልታት ተኸሉ", "ምስፍሕፋሕ ቫይረስ ንምንካይ ዝተበከሉ ተኽልታት ብቕጽበት ኣልዩን ኣጥፍኡን", "ንጻዕዳ ዝንቢ ንምክልኻል መብርሂ ዘለዎም መሸፈኒታት ተጠቐሙ", "ጽሬት፡ ንጻዕዳ ዝንቢ ወይ ንቫይረስ ከዕቁቡ ዝኽእሉ ኣእዋም ምቁጽጻር"], "chemical_control_header": "ምቁጽጻር ጻዕዳ ዝንቢ (ተሸካሚ):", "chemical_control_text": "ንጻዕዳ ዝንቢ ዒላማ ዝገበሩ ተባይ መከላኸሊ መድሃኒታት ንምሕደራ ኣገደስቲ እዮም። ኣብ እዋን ምትካል ዝውዕሉ ስነ-ስርዓታዊ ተባይ መከላኸሊ መድሃኒታት ወይ ናይ ቀጽሊ መረጻሕቲ ከድልዩ ይኽእሉ። ኣገባባት ስራሕ ተባይ መከላኸሊ መድሃኒታት ኣመሓይሹ። ንስልትታት ምሕደራ ጻዕዳ ዝንቢ ምስ ናይ ከባቢኹም ክኢላታት ተማኸሩ።", "further_info_link": ""}
    },
    "Tomato___Tomato_mosaic_virus": {
        "en": {"description": "Viral disease causing mosaic patterns and distortion on tomato leaves.", "symptoms_list": ["Light and dark green mottling (mosaic pattern) on leaves", "Leaf curling, distortion, stunting, and fern-like appearance of leaves", "Reduced fruit set and quality; fruit may show mottling or necrotic spots"], "cultural_control_header": "Management (No Cure for Infected Plants):", "cultural_control_list": ["Use virus-resistant tomato varieties (many are resistant to ToMV/TMV)", "Use certified disease-free seed and transplants", "Practice strict sanitation: wash hands thoroughly with soap after handling tobacco products or infected plants, as the virus is easily transmitted mechanically", "Avoid handling plants when wet", "Remove and destroy infected plants", "Control weeds that can host the virus"], "chemical_control_header": "Chemical Control:", "chemical_control_text": "No chemical treatments cure viral infections in plants. Focus is entirely on prevention and sanitation.", "further_info_link": ""},
        "am": {"description": "በቲማቲም ቅጠሎች ላይ የሞዛይክ ንድፎችን እና መዛባትን የሚያስከትል የቫይረስ በሽታ።", "symptoms_list": ["በቅጠሎች ላይ ቀላል እና ጥቁር አረንጓዴ ነጠብጣቦች (የሞዛይክ ንድፍ)", "የቅጠል መጠቅለል፣ መዛባት፣ እድገትን ማቆም እና የቅጠሎች የፈርን መሳይ መልክ", "የተቀነሰ የፍራፍሬ አቀማመጥ እና ጥራት፤ ፍራፍሬ ነጠብጣቦችን ወይም የኒክሮቲክ ነጠብጣቦችን ሊያሳይ ይችላል"], "cultural_control_header": "አያያዝ (ለተበከሉ ተክሎች ምንም ፈውስ የለም):", "cultural_control_list": ["ቫይረስን የሚቋቋሙ የቲማቲም ዝርያዎችን ይጠቀሙ (ብዙዎቹ ToMV/TMVን ይቋቋማሉ)", "የተረጋገጡ ከበሽታ ነጻ የሆኑ ዘሮችን እና ችግኞችን ይጠቀሙ", "ጥብቅ ንፅህናን ይለማመዱ፦ የትምባሆ ምርቶችን ወይም የተበከሉ ተክሎችን ከያዙ በኋላ እጅዎን በሳሙና በደንብ ይታጠቡ፣ ምክንያቱም ቫይረሱ በቀላሉ በሜካኒካዊ መንገድ ይተላለፋል", "እርጥብ በሚሆኑበት ጊዜ ተክሎችን ከመያዝ ይቆጠቡ", "የተበከሉ ተክሎችን ያስወግዱ እና ያጥፉ", "ቫይረሱን ሊያስተናግዱ የሚችሉ አረሞችን ይቆጣጠሩ"], "chemical_control_header": "የኬሚካል ቁጥጥር:", "chemical_control_text": "ምንም የኬሚካል ሕክምናዎች በእጽዋት ውስጥ የቫይረስ ኢንፌክሽኖችን አይፈውሱም። ትኩረቱ ሙሉ በሙሉ በመከላከል እና በንፅህና ላይ ነው።", "further_info_link": ""},
        "ti": {"description": "ቫይራዊ ሕማም ኣብ ቀጽሊ ቲማቲም ናይ ሞዛይክ ቅርጽታትን ምዝባዕን ዘስዕብ።", "symptoms_list": ["ኣብ ቀጸልቲ ፈኲስን ጸሊምን ቀጠልያ ነጠብጣባት (ናይ ሞዛይክ ቅርጺ)", "ምጭብጫብ ቀጽሊ፡ ምዝባዕ፡ ምድንጓይ ዕቤት፡ ከም ፈርን ዝመስል ትርኢት ቀጸልቲ", "ዝነከየ ምፍራይን ጽሬትን፤ ፍረ ነጠብጣባት ወይ ኒክሮቲክ ነጠብጣባት ከርኢ ይኽእል"], "cultural_control_header": "ምሕደራ (ንዝተበከሉ ተኽልታት ፈውሲ የብሎምን):", "cultural_control_list": ["ንቫይረስ ዝጻወሩ ዝርኣያት ቲማቲም ተጠቐሙ (ብዙሓት ንToMV/TMV ዝጻወሩ እዮም)", "ካብ ሕማም ነጻ ዝኾኑ ዘርእን ተኽልታትን ተጠቐሙ", "ጽኑዕ ጽሬት ተግበሩ፡ ድሕሪ ምሓዝ ፍርያት ትምባኾ ወይ ዝተበከሉ ተኽልታት ኣእዳውኩም ብሳሙና ጽቡቕ ጌርኩም ተሓጸቡ፡ ቫይረስ ብቐሊሉ ብሜካኒካዊ ኣገባብ ስለ ዝመሓላለፍ", "ተኽልታት ጥሉላት ኣብ ዝኾንሉ እዋን ካብ ምሓዝ ተቖጠቡ", "ዝተበከሉ ተኽልታት ኣልዩን ኣጥፍኡን", "ንቫይረስ ከዕቁቡ ዝኽእሉ ኣእዋም ምቁጽጻር"], "chemical_control_header": "ኬሚካላዊ ምቁጽጻር:", "chemical_control_text": "ዝኾነ ኬሚካላዊ ኣገባብ ኣብ ተኽልታት ንቫይራዊ ምትሕልላፍ ኣይፍውስን እዩ። ትኹረት ምሉእ ብምሉእ ኣብ ምክልኻልን ጽሬትን እዩ።", "further_info_link": ""}
    },
    "Tomato___healthy": {
        "en": {"description": "The tomato plant appears healthy.", "symptoms_list": ["No visible signs of targeted diseases."], "cultural_control_header": "General Tomato Care:", "cultural_control_list": ["Full sun, well-drained fertile soil", "Consistent watering, especially during fruiting", "Staking or caging for support", "Appropriate fertilization", "Pruning (suckering) for some varieties", "Monitor for common pests and diseases"], "chemical_control_header": "Preventative Measures:", "chemical_control_text": "Focus on cultural practices and resistant varieties. Fungicides may be needed preventatively for common foliar diseases if local pressure is high. Consult local guides.", "further_info_link": ""},
        "am": {"description": "የቲማቲም ተክል ጤናማ ይመስላል።", "symptoms_list": ["የተጠቆሙ በሽታዎች ምንም የሚታዩ ምልክቶች የሉም።"], "cultural_control_header": "አጠቃላይ የቲማቲም እንክብካቤ:", "cultural_control_list": ["ሙሉ ፀሐይ፣ ውሃ በደንብ የሚያሳልፍ ለም አፈር", "ወጥ የሆነ ውሃ ማጠጣት፣ በተለይም በፍራፍሬ ወቅት", "ለድጋፍ መደገፍ ወይም ማጠር", "ተገቢ ማዳበሪያ", "ለአንዳንድ ዝርያዎች መግረዝ (ማጥባት)", "የተለመዱ ተባዮችን እና በሽታዎችን መከታተል"], "chemical_control_header": "የመከላከያ እርምጃዎች:", "chemical_control_text": "በባህላዊ ልምዶች እና በሚቋቋሙ ዝርያዎች ላይ ያተኩሩ። የአካባቢ ግፊት ከፍተኛ ከሆነ ለተለመዱ የቅጠል በሽታዎች ፈንገስ መድኃኒቶች በመከላከል ሊያስፈልጉ ይችላሉ። የአካባቢ መመሪያዎችን ያማክሩ።", "further_info_link": ""},
        "ti": {"description": "እቲ ተኽሊ ቲማቲም ጥዑይ ይመስል።", "symptoms_list": ["ዝኾነ ዝርአ ምልክት ናይቶም ዝተጠቕሱ ሕማማት የብሉን።"], "cultural_control_header": "ሓፈሻዊ ክንክን ቲማቲም:", "cultural_control_list": ["ሙሉእ ጸሓይ፡ ማይ ጽቡቕ ዘንጠብጥብ ልሙዕ ሓመድ", "ቐጻሊ ምስታይ ማይ፡ ብፍላይ ኣብ እዋን ምፍራይ", "ንድጋፍ ምድጋፍ ወይ ምዕጻው", "ግቡእ ምዳበሪያ", "ንገሊኦም ዝርኣያት ምፍላጥ (ምጽባይ)", "ልሙዳት ተሃሳስን ሕማማትን ምቁጽጻር"], "chemical_control_header": "መከላኸሊ ስጉምትታት:", "chemical_control_text": "ኣብ ባህላዊ ተግባራትን ተጻወርቲ ዝርኣያትን ኣተኵሩ። ናይ ከባቢ ጸቕጢ ልዑል እንተኾይኑ፡ ንልሙዳት ሕማማት ቀጽሊ ከም መከላኸሊ ፈንገስ መከላኸሊ መድሃኒታት ከድልዩ ይኽእሉ። ናይ ከባቢኹም መምርሒታት ተወከሱ።", "further_info_link": ""}
    }
}

# --- Function to set background using Base64 ---
@st.cache(ttl=24*60*60, show_spinner=False) # Cache for 24 hrs for Streamlit 1.12.0
def get_base64_encoded_img_data(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        # st.warning(f"Background image file '{image_path}' not found.") # Optionally show warning in app
        print(f"Warning: Background image file '{image_path}' not found.") # Print to console
        return None

def set_page_background(image_file_path):
    if not os.path.exists(image_file_path):
        print(f"Warning: Background image '{image_file_path}' not found. Cannot set background.")
        return

    encoded_img_data = get_base64_encoded_img_data(image_file_path)
    if encoded_img_data:
        image_extension = image_file_path.split('.')[-1].lower()
        if image_extension not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            image_extension = 'jpeg'

        page_bg_img_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/{image_extension};base64,{encoded_img_data}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img_css, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'language' not in st.session_state:
    st.session_state.language = 'en' # Default to English language code
if 'active_tab_key' not in st.session_state:
    st.session_state.active_tab_key = "home_tab" # Use the key from TRANSLATIONS

# --- TRANSLATION FUNCTION ---
def _(text_key, **kwargs):
    # Fallback: current lang -> english -> key itself (if key not found even in English)
    current_lang_translations = TRANSLATIONS.get(st.session_state.language, TRANSLATIONS.get("en", {}))
    text_template = current_lang_translations.get(text_key, TRANSLATIONS.get("en", {}).get(text_key, f"KEY_NOT_FOUND: {text_key}"))
    try:
        return text_template.format(**kwargs) if kwargs else text_template
    except KeyError as e:
        # This happens if a {placeholder} in the string is not provided in kwargs
        print(f"Warning: Formatting key error for UI text key '{text_key}' in language '{st.session_state.language}': {e}. Using unformatted text.")
        return text_template # Return unformatted template on error

# --- SET BACKGROUND IMAGE ---
BG_IMAGE_FILE = "background.jpg"  
set_page_background(BG_IMAGE_FILE)
st.markdown("""
    <style>
    /* 1. Main Titles (Agricultural Gold) */
    h1, h2 {
        color: #F4D03F !important;
        text-shadow: 2px 2px 4px #000000 !important;
        font-weight: bold;
    }

    /* 2. Body Text (White with a strong shadow) */
    p, li, span, label, .stMarkdown {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px #000000 !important;
        font-size: 1.1rem !important;
    }

    /* 3. Give the text area a slight dark tint for better reading */
    .main .block-container {
        background-color: rgba(0, 0, 0, 0.3); /* Transparent black overlay */
        border-radius: 20px;
        padding: 40px !important;
    }

    /* 4. Make the buttons look like "Glass" */
    .stButton>button {
        color: #F4D03F !important;
        background-color: rgba(0, 0, 0, 0.6) !important;
        border: 2px solid #F4D03F !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_my_model():
    from huggingface_hub import hf_hub_download
    
    # 1. Provide your Hugging Face details
    # Replace 'tedd12t' with your actual HF username if it's different
    REPO_ID = "TeddyNigus/plant-disease-detector" 
    FILENAME = "trained_model.h5" 
    
    try:
        # 2. Download from Hugging Face
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        
        # 3. Load the model from the downloaded path
        model = tf.keras.models.load_model(model_path)
        return {"model": model}
    except Exception as e:
        # This keeps your existing error handling style
        return {"error": f"Error loading model from Hugging Face: {str(e)}"}

# --- Prediction Function ---
def model_prediction(test_image_uploader, model_data_dict_arg):
    if "error" in model_data_dict_arg or model_data_dict_arg.get("model") is None:
        # Error should be displayed by the caller based on model_data_dict_arg
        return None
    
    model = model_data_dict_arg["model"]
    try:
        image = tf.keras.preprocessing.image.load_img(test_image_uploader, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) # Convert single image to batch
        
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    except Exception as e:
        st.error(_("error_prediction_failed") + f" (Details: {str(e)})") # Show error in app
        return None
TAB_KEYS_ORDERED = ["home_page_option", "about_page_option", "disease_recognition_page_option"]
LANGUAGE_CODES_ORDERED = list(TRANSLATIONS.keys()) # ['en', 'am', 'ti']

# Initialize active_tab in session state if not present
if 'active_tab_key' not in st.session_state:
    st.session_state.active_tab_key = TAB_KEYS_ORDERED[0] # Default to home_tab

# Create columns for tab buttons and language switcher
num_navigation_buttons = len(TAB_KEYS_ORDERED) + 1  # Tabs + 1 language switch button
nav_columns = st.columns(num_navigation_buttons)

# Tab-like navigation buttons
for i, current_tab_key in enumerate(TAB_KEYS_ORDERED):
    if nav_columns[i].button(_(current_tab_key), key=f"button_nav_{current_tab_key}"):
        st.session_state.active_tab_key = current_tab_key
        st.experimental_rerun() # Rerun to update the active tab view immediately

# Language switcher button
current_language_code = st.session_state.language
current_language_index_in_list = LANGUAGE_CODES_ORDERED.index(current_language_code)
next_language_index = (current_language_index_in_list + 1) % len(LANGUAGE_CODES_ORDERED)
next_language_code_to_switch_to = LANGUAGE_CODES_ORDERED[next_language_index]

# Get the display name of the *next* language for the button text
# e.g., if current is 'en', next is 'am', button should say "Switch to Amharic"
next_language_display_name_key = f"lang_name_{next_language_code_to_switch_to}" # e.g., lang_name_am

if nav_columns[len(TAB_KEYS_ORDERED)].button(
    _("language_switch_button_text", next_lang_name=_(next_language_display_name_key)),
    key="button_language_switch"
):
    st.session_state.language = next_language_code_to_switch_to
    st.experimental_rerun() # Rerun to apply new language immediately

st.markdown("---") # Visual separator
model_data_dict_global = load_my_model()
if "error" in model_data_dict_global and st.session_state.active_tab_key == "recognition_tab":
    st.error(model_data_dict_global["error"]) # Error message is already translated


if st.session_state.active_tab_key == "home_page_option":
    st.header(_("home_header"))
    home_image_path = "home_page.jpeg"  # Ensure this file is in D:\plant\
    if os.path.exists(home_image_path):
        st.image(home_image_path, use_column_width=True)
    # else:
        # st.warning(f"Home page image '{home_image_path}' not found.") # Optional warning
    st.markdown(f"## { _('home_welcome_greeting') }") # Using ## for slightly smaller heading
    st.markdown(_("home_mission_statement"))
    st.markdown(f"### { _('home_how_it_works_header') }")
    st.markdown(_("home_step1_upload"))
    st.markdown(_("home_step2_analyze"))
    st.markdown(_("home_step3_results"))
    st.markdown(_("home_multilingual_note"))
    st.markdown("---")
    st.markdown(f"#### { _('home_call_to_action') }")

elif st.session_state.active_tab_key == "about_page_option":
    st.header(_("about_header"))
    st.subheader(_("about_introduction_header"))
    st.markdown(_("about_introduction_text"))
    st.subheader(_("about_technology_header"))
    st.markdown(_("about_tech_deep_learning"))
    st.markdown(_("about_tech_webapp"))
    st.markdown(_("about_tech_language"))
    st.subheader(_("about_dataset_header"))
    st.markdown(_("about_dataset_description"))
    st.subheader(_("about_features_header"))
    st.markdown(_("about_feature_prediction"))
    st.markdown(_("about_feature_info"))
    st.markdown(_("about_feature_multilingual"))
    st.subheader(_("about_developer_header"))
    # Remember to replace placeholders in TRANSLATIONS for this key
    st.markdown(_("about_developer_text"))
    st.subheader(_("about_future_scope_header"))
    st.markdown(_("about_future_scope_text"))

elif st.session_state.active_tab_key == "disease_recognition_page_option":
    st.header(_("recognition_header"))

    # Check again if model loaded properly before showing uploader
    if "error" in model_data_dict_global or model_data_dict_global.get("model") is None:
        if "error" not in model_data_dict_global: # If no specific error message from load_my_model
             st.error(_("error_model_load"))
    else: 
        uploaded_test_image = st.file_uploader(
    label=_("file_uploader_main_label"),   # This gets translated
    help=_("file_uploader_help_text"),     # This gets translated (tooltip)
    type=["jpg", "jpeg", "png"],
    key="widget_file_uploader_recognition" # Or your unique key
)
        if uploaded_test_image is not None:
            # 1. Create columns for side-by-side view
            col1, col2 = st.columns([1, 1]) 

            with col1:
                # Display image immediately
                st.image(uploaded_test_image, caption="Uploaded Image", use_column_width=True)

            with col2:
                # 2. RUN PREDICTION AUTOMATICALLY (No button needed)
                with st.spinner(_("spinner_text")):
                    prediction_result_index = model_prediction(uploaded_test_image, model_data_dict_global)

                if prediction_result_index is not None:
                            # IMPORTANT: This list MUST be accurate and in the model's output order
                         technical_class_names_from_model = [
                                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                'Tomato___healthy'
                            ]
                         if 0 <= prediction_result_index < len(technical_class_names_from_model):
                                predicted_technical_name = technical_class_names_from_model[prediction_result_index]

                                # Get the display name for the current language
                                disease_name_translations_current_lang = DISEASE_NAME_TRANSLATIONS.get(st.session_state.language, DISEASE_NAME_TRANSLATIONS.get("en", {}))
                                displayed_disease_name = disease_name_translations_current_lang.get(predicted_technical_name, predicted_technical_name) # Fallback to technical name

                                st.success(_("model_predict_msg", disease_name=displayed_disease_name))

                                # --- Displaying Recommendation ---
                                disease_specific_recommendations = DISEASE_RECOMMENDATIONS.get(predicted_technical_name, {})
                                recommendations_in_current_lang = disease_specific_recommendations.get(st.session_state.language, disease_specific_recommendations.get("en", {}))

                                if recommendations_in_current_lang:
                                    st.subheader(_("recommendations_subheader"))

                                    description = recommendations_in_current_lang.get("description")
                                    if description:
                                        st.markdown(f"**{_('description_label')}:** {description}")

                                    symptoms_list = recommendations_in_current_lang.get("symptoms_list")
                                    if symptoms_list:
                                        st.markdown(f"**{_('symptoms_label')}:**")
                                        if isinstance(symptoms_list, list):
                                            for item in symptoms_list:
                                                st.markdown(f"- {item}")
                                        else: # If it's a single string
                                            st.markdown(symptoms_list)

                                    cultural_header = recommendations_in_current_lang.get("cultural_control_header")
                                    cultural_list = recommendations_in_current_lang.get("cultural_control_list")
                                    if cultural_header and cultural_list:
                                        st.markdown(f"**{cultural_header}**")
                                        if isinstance(cultural_list, list):
                                            for item in cultural_list:
                                                st.markdown(f"- {item}")
                                        else:
                                            st.markdown(cultural_list)
                                    
                                    chemical_header = recommendations_in_current_lang.get("chemical_control_header")
                                    chemical_text = recommendations_in_current_lang.get("chemical_control_text")
                                    if chemical_header and chemical_text:
                                        st.markdown(f"**{chemical_header}**")
                                        st.markdown(chemical_text) # Use warning for emphasis on chemical use caution

                                    further_info_link = recommendations_in_current_lang.get("further_info_link", "")
                                    if further_info_link.strip() and not further_info_link.startswith("SEARCH_") and not further_info_link.startswith("PLACEHOLDER_"):
                                        st.markdown(f"[{_('further_info_label')}]({further_info_link})")
                                    
                                    st.markdown("---")
                                    #st.info(_("expert_consultation_disclaimer"))
                                else:
                                    st.info(_("no_recommendation_available"))
                         else:
                             st.error(_("error_prediction_index_range"))
        elif uploaded_test_image is None:
            # Only show this if the model is loaded but no image uploaded yet
            st.info(_("info_upload_image"))
