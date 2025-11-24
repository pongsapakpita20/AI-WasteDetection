# ------------------------------------
# ไฟล์: voice_guidance.py
# ------------------------------------
import os
import time
import threading
import queue
import subprocess
import sys

# --- 1. ฐานข้อมูลคำแนะนำการทิ้งขยะแบบเป็นธรรมชาติ ---
# Mapping ID -> ชื่อคลาส (ตามไฟล์ data.yaml)
CLASS_NAME_MAP = {
    0: "battery",
    1: "can",
    2: "cardboard_bowl",
    3: "cardboard_box",
    4: "chemical_plastic_bottle",
    5: "chemical_plastic_gallon",
    6: "chemical_spray_can",
    7: "light_bulb",
    8: "paint_bucket",
    9: "plastic_bag",
    10: "plastic_bottle",
    11: "plastic_bottle_cap",
    12: "plastic_box",
    13: "plastic_cultery",
    14: "plastic_cup",
    15: "plastic_cup_lid",
    16: "reuseable_paper",
    17: "scrap_paper",
    18: "scrap_plastic",
    19: "snack_bag",
    20: "stick",
    21: "straw",
}

FRIENDLY_MESSAGE_MAP = {
    "battery": "This is a battery. Please dispose in hazardous waste bin and tape the terminals first.",
    "can": "This is a beverage can. Please rinse clean and recycle.",
    "cardboard_bowl": "This is a cardboard bowl. Please remove food residue and recycle.",
    "cardboard_box": "This is a cardboard box. Please flatten and bundle before recycling.",
    "chemical_plastic_bottle": "This is a chemical plastic bottle. Please seal tightly and dispose at hazardous waste collection point.",
    "chemical_plastic_gallon": "This is a chemical plastic gallon. Please rinse and send to hazardous waste disposal center.",
    "chemical_spray_can": "This is a chemical spray can. Please puncture to release pressure and dispose in hazardous waste bin.",
    "light_bulb": "This is a light bulb. Please wrap in paper before disposing in hazardous waste bin.",
    "paint_bucket": "This is a paint bucket. Please let paint dry and dispose at waste disposal center.",
    "plastic_bag": "This is a plastic bag. Please reuse or dispose in general waste bin.",
    "plastic_bottle": "This is a plastic bottle. Please rinse and recycle.",
    "plastic_bottle_cap": "This is a plastic bottle cap. Please separate from bottle and recycle.",
    "plastic_box": "This is a plastic box. Please rinse clean and recycle.",
    "plastic_cultery": "This is plastic cutlery. Please dispose in general waste or reuse when possible.",
    "plastic_cup": "This is a plastic cup. Please rinse and recycle.",
    "plastic_cup_lid": "This is a plastic cup lid. Please wipe dry before recycling.",
    "reuseable_paper": "This is reusable paper. Please recycle.",
    "scrap_paper": "This is scrap paper. Please recycle.",
    "scrap_plastic": "This is scrap plastic. Please collect and recycle.",
    "snack_bag": "This is a snack bag. Please shake out food residue and recycle in mixed packaging bin.",
    "stick": "This is a stick or popsicle stick. Please dispose in general waste or compost.",
    "straw": "This is a plastic straw. Please dispose in general waste or use your own reusable straw.",
}

DEFAULT_MESSAGE = "Unknown waste item. Please check again."

def get_guidance_text(class_id: int) -> str:
    """
    แปลง class_id ให้กลายเป็นข้อความคำแนะนำที่เป็นธรรมชาติ
    """
    class_name = CLASS_NAME_MAP.get(class_id)
    if not class_name:
        return DEFAULT_MESSAGE
    return FRIENDLY_MESSAGE_MAP.get(class_name, f"ตรวจพบ {class_name} กรุณาจัดการอย่างถูกถังครับ")

# --- 2. ฟังก์ชันพูดด้วย PowerShell (ใช้ Windows SAPI) ---
def _speak_with_powershell(text: str):
    """
    พูดด้วย PowerShell โดยใช้ Windows SAPI
    วิธีนี้หลีกเลี่ยงปัญหา thread safety และ run loop ของ pyttsx3
    """
    try:
        # Escape ข้อความสำหรับ PowerShell (ใช้ single quote เพื่อหลีกเลี่ยงปัญหา escape)
        # แทนที่ single quote ด้วย double single quote
        escaped_text = text.replace("'", "''")
        
        # สร้าง PowerShell command
        ps_command = f'''
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.Rate = 0
$speak.Volume = 100
$speak.Speak('{escaped_text}')
'''
        
        # รัน PowerShell command
        result = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        if result.returncode != 0:
            print(f"[VOICE][powershell] error (code {result.returncode}): {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[VOICE][powershell] timeout เกิน 15 วินาที")
        return False
    except Exception as ex:
        print(f"[VOICE][powershell] error: {ex}")
        import traceback
        traceback.print_exc()
        return False


# --- 3. Queue และ Worker Thread สำหรับการพูด ---
speech_queue = queue.Queue()
speech_worker_running = False
speech_worker_thread = None

def _speech_worker():
    """
    Worker thread ที่รอรับข้อความจาก queue และพูดทีละข้อความ
    ใช้ PowerShell แทน pyttsx3 เพื่อหลีกเลี่ยงปัญหา thread safety
    """
    global speech_worker_running
    
    try:
        print("[VOICE][worker] กำลังเริ่มต้น worker thread...")
        print("[VOICE][worker] Worker thread เริ่มทำงาน - กำลังรอรับข้อความ...")
        
        while speech_worker_running:
            try:
                # รอรับข้อความจาก queue (timeout 1 วินาที)
                text = speech_queue.get(timeout=1)
                if text is None:  # Signal to stop
                    print("[VOICE][worker] ได้รับสัญญาณหยุด")
                    break
                
                queue_size_before = speech_queue.qsize()
                print(f"[VOICE][worker] ได้รับข้อความ: '{text[:50]}...' (queue size: {queue_size_before})")
                
                # พูดด้วย PowerShell (จะพูดจนเสร็จแม้ไม่มี detection ต่อ)
                print(f"[VOICE][worker] กำลังพูดด้วย PowerShell...")
                success = _speak_with_powershell(text)
                
                if success:
                    print(f"[VOICE][worker] เสร็จสิ้น (queue size ตอนนี้: {speech_queue.qsize()})")
                else:
                    print(f"[VOICE][worker] WARNING: การพูดล้มเหลว")
                
                speech_queue.task_done()
            except queue.Empty:
                # Timeout - วนลูปต่อ (ไม่พิมพ์ log เพื่อลด noise)
                continue
            except Exception as ex:
                print(f"[VOICE][worker] error ระหว่างการพูด: {ex}")
                import traceback
                traceback.print_exc()
                try:
                    speech_queue.task_done()
                except:
                    pass
    except Exception as ex:
        print(f"[VOICE][worker] initialization error: {ex}")
        import traceback
        traceback.print_exc()
    finally:
        print("[VOICE][worker] Worker thread หยุดทำงาน")
        speech_worker_running = False

def _start_speech_worker():
    """เริ่ม worker thread สำหรับการพูด"""
    global speech_worker_running, speech_worker_thread
    
    if not speech_worker_running:
        print("[VOICE] กำลังเริ่ม worker thread...")
        speech_worker_running = True
        speech_worker_thread = threading.Thread(target=_speech_worker, daemon=True)
        speech_worker_thread.start()
        print("[VOICE] Speech worker thread เริ่มทำงาน")
        
        # ตรวจสอบว่า thread ทำงานจริงหรือไม่ (รอ 0.5 วินาที)
        time.sleep(0.5)
        if not speech_worker_thread.is_alive():
            print("[VOICE] WARNING: Worker thread ไม่ทำงาน! กำลังลองเริ่มใหม่...")
            speech_worker_running = False
            speech_worker_thread = threading.Thread(target=_speech_worker, daemon=True)
            speech_worker_thread.start()
            time.sleep(0.5)
            if speech_worker_thread.is_alive():
                print("[VOICE] Worker thread เริ่มทำงานสำเร็จ")
            else:
                print("[VOICE] ERROR: Worker thread ยังไม่ทำงาน!")
        else:
            print("[VOICE] Worker thread ทำงานปกติ")

# เริ่ม worker thread ทันทีเมื่อ import module
_start_speech_worker()

# --- 4. ฟังก์ชันพูด (พร้อม Debounce) ---
last_spoken_time = 0
last_spoken_class = -1
DEBOUNCE_TIME_SECONDS = 3 # หน่วงเวลา 3 วินาที ก่อนพูดซ้ำ

def speak_guidance(class_id):
    """
    พูดคำแนะนำสำหรับ class_id ที่ระบุ
    พร้อมระบบ Debounce 3 วินาทีสำหรับคลาสเดิม
    
    พฤติกรรม:
    - ถ้ามีขยะใหม่ (คลาสใหม่) → พูดทันที (เข้า queue)
    - ถ้ากำลังพูดอยู่ → queue จะพูดต่อกันไปตามลำดับ
    - ถ้าเป็นคลาสเดิม → รอ Debounce (3 วินาที)
    """
    global last_spoken_time, last_spoken_class

    current_time = time.time()
    
    # ตรวจสอบว่า (ไม่ใช่คลาสเดิม) หรือ (เป็นคลาสเดิม แต่พูดไปนานแล้ว)
    if (class_id != last_spoken_class) or \
       (current_time - last_spoken_time > DEBOUNCE_TIME_SECONDS):
        
        # 1. หาคำแนะนำจาก ID
        text_to_speak = get_guidance_text(class_id)
        
        queue_size_before = speech_queue.qsize()
        print(f"[VOICE] กำลังส่งข้อความเข้า queue: (คลาส {class_id}) '{text_to_speak[:50]}...' (queue size: {queue_size_before})")
        
        # 2. ส่งข้อความเข้า queue เพื่อให้ worker thread พูด
        # หมายเหตุ: ถ้ากำลังพูดอยู่ ข้อความใหม่จะเข้า queue และพูดต่อกันไปตามลำดับ
        _queue_speech(text_to_speak)
        
        # 3. อัปเดตเวลาและคลาสล่าสุด
        last_spoken_time = current_time
        last_spoken_class = class_id
    else:
        # ยังไม่ผ่าน Debounce - ไม่พูดซ้ำ
        time_since_last = current_time - last_spoken_time
        print(f"[VOICE] ข้ามการพูด (คลาส {class_id} เดียวกัน, ผ่านไป {time_since_last:.1f} วินาที, ต้องรอ {DEBOUNCE_TIME_SECONDS} วินาที)")


def _queue_speech(text: str):
    """
    ส่งข้อความเข้า queue เพื่อให้ worker thread พูดด้วย PowerShell
    """
    try:
        # ตรวจสอบว่า worker thread ทำงานอยู่หรือไม่
        if not speech_worker_running:
            _start_speech_worker()
        
        # ส่งข้อความเข้า queue
        speech_queue.put(text)
        print(f"[VOICE] ส่งข้อความเข้า queue: '{text}'")
    except Exception as ex:
        print(f"[VOICE] error: {ex}")