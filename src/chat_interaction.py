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
                f"{context_str}"
                f"Anda adalah seorang dokter yang memberikan informasi tentang penyakit {disease_context} kepada pasien penderita penyakit tersebut. "
                "Berikan penjelasan berdasarkan konteks yang ada dan pengetahuan medis Anda."
                "Jelaskan dengan bahasa yang mudah dipahami."
                "Tidak perlu memperkenalkan diri Anda"
            )
        else:
            system_prompt = (
                f"{context_str}"
                "Anda adalah seorang dokter yang sedang melakukan konsultasi. "
                "Tanyakan gejala-gejala yang dirasakan pasien secara bertahap. "
                "Berikan respon dan analisis untuk setiap jawaban pasien. "
                "Fokus pada pengumpulan informasi yang relevan untuk diagnosis. "
                "Selalu tanyakan apakah pasien memiliki gejala lain di kalimat terakhir."
            )

        if not messages:
            chat_history.add_message(SystemMessage(content=system_prompt))
        
        chat_history.add_message(HumanMessage(content=human_messages))

        if not diagnosis:
            extraction_prompt = (
                "Analisis pesan pasien berikut dan ekstrak apabila terdapat keluhan medis atau gejala dalam format JSON:\n"
                "Contohnya:\n"
                "{\n"
                '  "gejala": ["pusing", "batuk", "lemas"]\n'
                "}\n\n"
                "Apabila tidak ada gejala, isi dengan JSON { \"gejala\": []}\n\n"
                "Hanya jawab dengan format JSON."
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

def check_if_chat_is_symptoms(human_messages: str, model: str) -> bool:
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
        system_prompt = (
            "Anda adalah seorang dokter yang ahli dalam mendeteksi apakah seseorang sedang "
            "membicarakan tentang gejala penyakit atau kondisi kesehatan mereka. "
            "Tugas Anda adalah menentukan apakah pesan yang diberikan berisi informasi "
            "tentang gejala, keluhan kesehatan, atau pertanyaan medis.\n\n"
            "Berikan jawaban 'true' jika pesan berisi informasi tentang gejala atau "
            "keluhan kesehatan, dan 'false' jika tidak."
        )
        
        user_prompt = f"Pesan: {human_messages}\n\nApakah pesan di atas membicarakan tentang gejala atau keluhan kesehatan?"
        
        if model.lower() == 'groq_llama3_70b':
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content.lower().strip()
            
        elif model.lower() == 'diffbot':
            # For Diffbot API
            prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm.complete(
                prompt=prompt,
                temperature=0,
                max_tokens=10
            )
            result = response.lower().strip()
        
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Check if response indicates symptoms
        return 'true' in result or 'ya' in result or 'benar' in result
        
    except Exception as e:
        logging.error(f"Error in chat_interaction: {str(e)}")
        raise Exception(f"Failed to process chat interaction: {str(e)}")

print(chat_interaction("groq_llama3_70b", "Itu aja sih dok. Saya sakit apa ya?", "1", {"name": "Budi", "age": 30, "weight": 70, "height": 170, "description": "Ada riwayat diabetes"}, True, "Pasien kemungkinan menderita diabetes. Diabetes adalah penyakit yang disebabkan oleh kadar gula darah yang tinggi. Obatnya dengan minum air putih dan kurangi gula."))