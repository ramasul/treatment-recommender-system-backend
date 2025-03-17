import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from src.QA_integration import get_history_by_session_id, get_total_tokens
from src.llm import get_llm

logging.basicConfig(format='%(asctime)s - %(message)s',level='INFO')

def chat_interaction(
    model: str,
    human_messages: str,
    session_id: str,
    context: Optional[Dict] = None,
    diagnosis: bool = False,
    disease_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    [ENG]: Handle medical chat interactions with diagnosis and informational modes.
    [IDN]: Mengelola interaksi obrolan medis dengan mode diagnosis dan informasi.

    Args:
        model: LLM model instance (Groq or Diffbot)
        human_messages: Current user message
        session_id: Unique session identifier for chat history
        context: User context including age, name, weight, height (JSON)
        diagnosis: Boolean flag for diagnosis mode
        disease_context: Specific disease context for informational mode
        is_initial: bool = False
    
    Returns:
        Dict containing response, symptoms (if applicable), and chat metadata
    """
    try:
        llm, model_name = get_llm(model)
        chat_history = get_history_by_session_id(session_id)
        messages = chat_history.messages

        context_str = ""
        if context:
            context_str = (
                f"Informasi Pasien:\n"
                f"Nama: {context.get('name', 'Tidak disebutkan')}\n"
                f"Usia: {context.get('age', 'Tidak disebutkan')}\n"
                f"Berat Badan: {context.get('weight', 'Tidak disebutkan')} kg\n"
                f"Tinggi Badan: {context.get('height', 'Tidak disebutkan')} cm\n"
                f"Deskripsi: {context.get('description', 'Tidak ada deskripsi tambahan')}\n\n"
            )

        if diagnosis:
            system_prompt = (
                f"Anda adalah seorang dokter yang memberikan informasi kepada pasien penderita penyakit. "
                f"{context_str}"
                "Berikan penjelasan berdasarkan konteks penyakit pasien yang ada."
                "Apabila tidak ada konteks yang relevan, berikan informasi umum dan jangan membuat informasi baru."
                "Jelaskan dengan bahasa yang mudah dipahami."
                "Jangan lakukan penanganan, hanya berikan informasi."
                "Tidak perlu memperkenalkan diri Anda."
                "Jawab dalam maksimum 2 kalimat."
                "Jawab dalam bahasa Indonesia."
                f"Penyakit pasien: {disease_context}"
            )
        else:
            system_prompt = (
                f"Ada seorang pasien dengan, {context_str}"
                "Anda adalah seorang dokter yang sedang memberikan konsultasi."
                "Selalu tanyakan gejala lain yang dirasakan pasien."
                "Fokus pada pengumpulan informasi yang relevan untuk diagnosis."
                "Jangan sebut nama penyakit apapun."
                "Jawab dalam kurang dari 14 kata."
            )

        if not messages:
            chat_history.add_message(SystemMessage(content=system_prompt))
        
        chat_history.add_message(HumanMessage(content=human_messages))

        if not diagnosis:
            # Get previous AI message if it exists
            previous_ai_message = None
            if len(messages) >= 2 and isinstance(messages[-1], HumanMessage) and isinstance(messages[-2], AIMessage):
                previous_ai_message = messages[-2].content
            
            extraction_prompt = (
                "Analisis pesan pasien berikut dan ekstrak apabila terdapat keluhan medis atau gejala dalam format JSON.\n"
                
                # Add context from previous AI message if available
                f"{'Pertanyaan sebelumnya dari dokter: ' + previous_ai_message if previous_ai_message else ''}\n\n"
                
                "Berikan hasil dalam format JSON:\n"
                "{\n"
                '  "gejala": ["pusing", "batuk", "lemas"]\n'
                "}\n\n"

                "Contoh:\n"
                "Pertanyaan: 'Apakah Anda merasa pusing atau mual?'\n"
                "Pesan Pasien: 'Engga, tetapi saya kesulitan ereksi'\n"
                "Jawaban: { \"gejala\": [\"kesulitan ereksi\"]}\n\n"

                "Contoh:\n"
                "Pertanyaan: 'Apakah Anda merasa pusing atau mual?'\n"
                "Pesan Pasien: 'Iya'\n"
                "Jawaban: { \"gejala\": [\"pusing\", \"mual\"]}\n\n"
                
                "Catatan penting:\n"
                "- Jika pasien menjawab 'ya', 'iya', 'ada', 'betul', dll. terhadap pertanyaan tentang gejala tertentu, ekstrak gejala tersebut\n"
                "- Gunakan konteks dari pertanyaan sebelumnya untuk memahami jawaban pasien yang singkat\n"
                "- Apabila tidak ada gejala, isi dengan { \"gejala\": []}\n\n"
                
                "Hanya jawab dengan format JSON.\n"
                f"Pesan pasien: {human_messages}"
            )
            
            symptom_response = llm.invoke([HumanMessage(content=extraction_prompt)])
            print(symptom_response)
            
            if isinstance(symptom_response, AIMessage):
                extracted_symptoms = symptom_response.content.strip()
            else:
                extracted_symptoms = "{}"

        # Get llm response for the main conversation
        chat_response = llm.invoke(messages)
        total_tokens = get_total_tokens(chat_response, llm)
        
        # Process symptoms if in consultation mode
        symptoms_summary = None
        if not diagnosis and extracted_symptoms:
            try:
                symptoms_summary = json.loads(extracted_symptoms)
            except json.JSONDecodeError:
                logging.warning("Failed to parse symptom extraction JSON")
                symptoms_summary = {
                    "gejala": [],
                }

        # Save AI response to history
        chat_history.add_message(AIMessage(content=chat_response.content))

        return {
            "session_id": session_id,
            "message": chat_response.content,
            "symptoms_summary": symptoms_summary,
            "timestamp": datetime.now().isoformat(),
            "info": {
                "model": model_name,
                "total_tokens": total_tokens,
                "response_time": 0,
            },
            "user": "chatbot"
        }

    except Exception as e:
        logging.error(f"Error in chat_interaction: {str(e)}")
        raise Exception(f"Failed to process chat interaction: {str(e)}")
    
def initial_greeting(session_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    [ENG]: Generate the initial greeting message for medical consultation.
    [IDN]: Menghasilkan pesan sambutan awal untuk konsultasi medis.
    
    Args:
        session_id: Unique session identifier
        context: Optional dictionary containing patient information
        
    Returns:
        Dict containing the greeting response
    """
    patient_name = context.get('name', 'Bapak/Ibu') if context else 'Bapak/Ibu'
    greeting_prompt = (
        f"Selamat datang {patient_name}, saya Dr. DTETI. "
        "Apa yang bisa saya bantu? "
        "Mohon ceritakan keluhan yang Anda rasakan saat ini."
    )
    
    return {
        "session_id": session_id,
        "message": greeting_prompt,
        "info": {
                "model": "-",
                "total_tokens": 0,
                "response_time": 0,
            },
        "timestamp": datetime.now().isoformat(),
        "user": "chatbot"
    }

def check_if_chat_is_symptoms(human_messages: str, model: str, session_id: str) -> bool:
    """
    [ENG]: Check if the chat message contains symptoms extraction request.
    [IDN]: Periksa apakah pesan obrolan berisi permintaan ekstraksi gejala.
   
    Args:
        human_messages: User chat message
        model: Model name/identifier for the LLM
       
    Returns:
        Boolean flag indicating if the chat message is a symptom extraction request
    """
    try:
        llm, model_name = get_llm(model)
        
        # Get chat history using session_id
        chat_history = get_history_by_session_id(session_id)
        messages = chat_history.messages
        
        # Get previous AI message if it exists
        previous_ai_message = None
        if len(messages) >= 1 and isinstance(messages[-1], HumanMessage):
            # Find the most recent AI message before the current human message
            for i in range(len(messages)-2, -1, -1):
                if isinstance(messages[i], AIMessage):
                    previous_ai_message = messages[i].content
                    break
        
        system_prompt = (
            "Anda adalah seorang dokter yang ahli dalam mendeteksi apakah seseorang sedang "
            "membicarakan tentang gejala penyakit atau kondisi kesehatan mereka. "
            "Tugas Anda adalah menentukan apakah pesan yang diberikan berisi informasi "
            "tentang gejala, keluhan kesehatan, atau pertanyaan medis.\n\n"
            "PENTING: Jika dokter menanyakan tentang gejala dan pasien menjawab APAPUN yang "
            "mengonfirmasi gejala tersebut (seperti 'ya', 'iya', 'betul', 'ada', dll.) "
            "atau memberikan durasi/detail tentang gejala tersebut, maka ini "
            "harus dianggap sebagai pembicaraan tentang gejala.\n\n"
            "Berikan jawaban 'true' jika pesan berisi informasi tentang gejala atau "
            "keluhan kesehatan, atau jika pesan adalah respons terhadap pertanyaan dokter "
            "tentang gejala. Jawab 'false' jika tidak terkait dengan gejala atau keluhan kesehatan."
        )
        
        # Include previous AI message for context if available
        context = ""
        if previous_ai_message:
            context = f"Pertanyaan dokter sebelumnya: {previous_ai_message}\n\n"
        
        user_prompt = (
            f"{context}"
            f"Pesan pasien: {human_messages}\n\n"
            f"Berdasarkan percakapan di atas, apakah pasien sedang membicarakan atau "
            f"merespons tentang gejala/keluhan kesehatan? Jawab hanya dengan 'true' atau 'false'."
        )
       
        if model.lower() == 'groq_llama3_70b' or 'openai' in model.lower():
            from langchain_core.messages import SystemMessage, HumanMessage
           
            prompt_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
           
            response = llm.invoke(prompt_messages)
            result = response.content.lower().strip()
        else:
            raise ValueError(f"Unsupported model: {model}")
       
        # More comprehensive check of positive responses
        positive_indicators = ['true', 'ya', 'benar', 'iya', 'betul']
        
        # Log the result for debugging
        logging.info(f"Symptom detection for '{human_messages}' with previous context '{previous_ai_message}': {result}")
        
        # Check if any positive indicator is in the result
        return any(indicator in result for indicator in positive_indicators)
       
    except Exception as e:
        logging.error(f"Error in check_if_chat_is_symptoms: {str(e)}")
        # Default to True in case of error to be safe
        return True

# print(chat_interaction("groq_llama3_70b", "Akhir-akhir ini saya ngerasa kepala sering nggeliyeng kepala serasa muter muter gitu dok.", "1", {"name": "Budi", "age": 30, "weight": 70, "height": 170, "description": "Ada riwayat diabetes"}, False, None))