# ------------------------------------
# ไฟล์: app.py
# ------------------------------------
import gradio as gr
import cv2
from ultralytics import YOLO
from voice_guidance import speak_guidance, CLASS_NAME_MAP  # Import ฟังก์ชันพูดและ class names
import threading
import time
import os
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------------
# (สำคัญ!) แก้ไข Path นี้ให้ตรงกับไฟล์ best.pt ที่คุณเทรนได้
# -------------------------------------------------------------------
MODEL_PATH = 'artifacts/models/waste-sorter-best.pt' # ใช้โมเดลที่ promote แล้วจาก DVC pipeline
# -------------------------------------------------------------------

# 1. โหลดโมเดล AI ที่เทรนเสร็จแล้ว
try:
    print(f"กำลังโหลดโมเดลจาก: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("โหลดโมเดลสำเร็จ")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    print("กรุณาตรวจสอบว่า Path ของโมเดลถูกต้องหรือไม่")
    exit()

# การตั้งค่าเพิ่มเติมสำหรับระบบเสียง
SPEECH_CONF_THRESHOLD = 0.4          # conf ขั้นต่ำที่จะพิจารณาพูด (ลดเพื่อให้พูดง่ายขึ้น)
SUSTAINED_FRAME_THRESHOLD = 2        # จำนวนเฟรมต่อเนื่องก่อนพูด (ลดความเข้มงวด)
ANNOUNCE_COOLDOWN_SECONDS = 6        # เวลาระหว่างการพูดซ้ำคลาสเดิม

# การตั้งค่าสำหรับบันทึกภาพ
SAVE_IMAGES = True                   # เปิด/ปิดการบันทึกภาพ
SAVE_CONF_THRESHOLD = 0.5            # conf ขั้นต่ำที่จะบันทึกภาพ
SAVE_COOLDOWN_SECONDS = 3            # เวลาระหว่างการบันทึกภาพซ้ำ (วินาที)
SAVE_DIR = "detected_waste"          # โฟลเดอร์สำหรับเก็บภาพ

# ตัวแปรสถานะสำหรับระบบเสียง
last_detected_class_for_speech = -1
current_streak_class = -1
current_streak_length = 0
last_announced_class = -1
last_announced_time = 0.0

# ตัวแปรสถานะสำหรับบันทึกภาพ
last_saved_class = -1
last_saved_time = 0.0

def save_detected_image(frame, class_id, confidence):
    """
    บันทึกภาพที่ตรวจจับได้ไปยังโฟลเดอร์ตามประเภทขยะ
    """
    global last_saved_class, last_saved_time
    
    if not SAVE_IMAGES:
        return
    
    # ตรวจสอบ cooldown
    now = time.time()
    if class_id == last_saved_class and (now - last_saved_time) < SAVE_COOLDOWN_SECONDS:
        return
    
    # ตรวจสอบ confidence threshold
    if confidence < SAVE_CONF_THRESHOLD:
        return
    
    try:
        # ดึงชื่อคลาส
        class_name = CLASS_NAME_MAP.get(class_id, f"unknown_{class_id}")
        
        # สร้างโฟลเดอร์ตามประเภทขยะ
        class_dir = Path(SAVE_DIR) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # สร้างชื่อไฟล์: timestamp_class_conf.jpg
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        filename = f"{timestamp}_{class_name}_{confidence:.2f}.jpg"
        filepath = class_dir / filename
        
        # บันทึกภาพ
        cv2.imwrite(str(filepath), frame)
        
        # อัปเดตสถานะ
        last_saved_class = class_id
        last_saved_time = now
        
        print(f"[SAVE] Saved: {filepath}")
        
    except Exception as e:
        print(f"[SAVE] Error saving image: {e}")

def run_speech_in_background():
    """
    ฟังก์ชันนี้จะรันใน Thread แยก
    เพื่อเรียกใช้ speak_guidance โดยไม่ทำให้วิดีโอค้าง
    """
    global last_detected_class_for_speech
    while True:
        if last_detected_class_for_speech != -1:
            # เก็บค่าไว้ก่อน แล้วรีเซ็ตทันที
            class_to_speak = last_detected_class_for_speech
            last_detected_class_for_speech = -1
            
            # เรียกใช้ฟังก์ชันพูด (ซึ่งมี Debounce ของตัวเอง)
            speak_guidance(class_to_speak)
        # ป้องกัน busy loop
        time.sleep(0.05)

def process_frame(frame):
    """
    ฟังก์ชันหลักที่ Gradio จะเรียกใช้สำหรับทุกเฟรมจาก Webcam
    """
    global last_detected_class_for_speech, current_streak_class, current_streak_length
    global last_announced_class, last_announced_time
    
    # 1. พลิกเฟรม (กล้อง Webcam มักจะกลับด้าน)
    frame = cv2.flip(frame, 1)
    # แปลงเป็น BGR สำหรับโมเดล (Gradio ป้อน RGB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 2. สั่งให้โมเดลตรวจจับวัตถุในเฟรม
    results = model(frame_bgr, conf=0.25, imgsz=640, verbose=False, max_det=300)
    
    # 3. วาดกรอบและชื่อคลาสลงบนภาพ (ฟังก์ชัน .plot() ของ ultralytics)
    annotated_bgr = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    try:
        print("boxes:", len(results[0].boxes))
    except Exception:
        pass
    
    # 4. ตรวจสอบว่าเจออะไรหรือไม่
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        detected_class_tensor = boxes.cls[0]
        detected_class = int(detected_class_tensor.item())
        detected_conf = float(boxes.conf[0].item()) if boxes.conf is not None else 0.0

        if detected_conf >= SPEECH_CONF_THRESHOLD:
            if detected_class == current_streak_class:
                current_streak_length += 1
            else:
                current_streak_class = detected_class
                current_streak_length = 1

            now = time.time()
            should_trigger_speech = (
                current_streak_length >= SUSTAINED_FRAME_THRESHOLD and
                (
                    detected_class != last_announced_class or
                    now - last_announced_time >= ANNOUNCE_COOLDOWN_SECONDS
                )
            )

            if should_trigger_speech:
                # ส่งคำสั่งให้พูด (เสียงจะพูดจนเสร็จแม้ไม่มี detection ต่อ)
                last_detected_class_for_speech = detected_class
                last_announced_class = detected_class
                last_announced_time = now
                
                # บันทึกภาพที่ตรวจจับได้ (ใช้ annotated_bgr ที่มีกรอบแล้ว)
                save_detected_image(annotated_bgr, detected_class, detected_conf)
                
                try:
                    print(f"[SPEECH] trigger class={detected_class} conf={detected_conf:.2f}")
                except Exception:
                    pass
        else:
            # conf ต่ำเกินไป - reset streak แต่ไม่หยุดเสียงที่กำลังพูดอยู่
            current_streak_class = -1
            current_streak_length = 0
    else:
        # ไม่เจอวัตถุ - reset streak แต่ไม่หยุดเสียงที่กำลังพูดอยู่
        # หมายเหตุ: เสียงที่ส่งเข้า queue แล้วจะพูดจนเสร็จ ไม่ว่าจะมี detection ต่อหรือไม่
        current_streak_class = -1
        current_streak_length = 0

    # คืนค่าภาพที่มีกรอบวาดแล้ว กลับไปแสดงที่หน้าเว็บ
    return annotated_frame

# --- สร้าง Gradio Interface ---
def main():
    print("กำลังสร้าง Gradio Interface...")
    
    # สร้างโฟลเดอร์สำหรับเก็บภาพ
    if SAVE_IMAGES:
        save_path = Path(SAVE_DIR)
        save_path.mkdir(exist_ok=True)
        print(f"บันทึกภาพไปที่: {save_path.absolute()}")
    
    # เริ่ม Thread สำหรับการพูดแยกต่างหาก
    speech_thread = threading.Thread(target=run_speech_in_background, daemon=True)
    speech_thread.start()

    # สร้างหน้าเว็บด้วย Blocks เพื่อควบคุม UI ได้มากขึ้น
    with gr.Blocks(title="ระบบคัดแยกขยะอัจฉริยะ", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 ระบบคัดแยกขยะอัจฉริยะ (AI Waste Sorter)")
        gr.Markdown("โครงงานโดย: พงศภัค, กฤติน, ภูริชทัต (ใช้ YOLOv12)")
        
        with gr.Row():
            input_image = gr.Image(
                type="numpy",
                sources=["webcam"],
                streaming=True,
                label="จ่อขยะที่กล้องนี้",
                show_label=True,
                show_download_button=False,
                show_share_button=False,
            )
            output_image = gr.Image(
                type="numpy",
                label="ผลการตรวจจับ",
                show_label=True,
                show_download_button=False,
                show_share_button=False,
            )
        
        # ใช้ streaming event สำหรับ real-time processing (ไม่มีปุ่ม Clear/Flag)
        input_image.stream(
            fn=process_frame,
            inputs=input_image,
            outputs=output_image,
        )
    
    # รันแอป
    print("Interface พร้อมใช้งาน. เปิดในเบราว์เซอร์ของคุณ...")
    demo.launch(share=False)  # share=True ถ้าต้องการส่งลิงก์ให้คนอื่นดู

if __name__ == '__main__':
    main()