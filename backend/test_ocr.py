import sys
import os
import base64

# Thêm thư mục backend vào sys.path để import ocr_service
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.ocr_service import extract_text_from_image

def main():
    # Hardcode đường dẫn ảnh tại đây
    # Thay đổi đường dẫn này thành đường dẫn ảnh thực tế trên máy bạn
    image_path = r"C:\Users\Hung\Desktop\VScodeproject\LLM_RAG_projects\PhishGuard-Multimodal-Phishing-Detection-System\backend\mail.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file ảnh tại: {image_path}")
        print("Vui lòng thay đổi biến 'image_path' trong code bằng đường dẫn ảnh hợp lệ.")
        return

    print(f"Đang đọc ảnh từ: {image_path}")
    
    try:
        # Đọc file ảnh và chuyển sang chuỗi base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        print("Đang tiến hành nhận dạng OCR (vui lòng đợi vài giây)...")
        # Gọi hàm xử lý hiện tại của pipeline
        result = extract_text_from_image(encoded_string)
        
        if result.get("error"):
            print(f"❌ Lỗi khi OCR: {result['error']}")
        elif result.get("text"):
            print("\n" + "="*50)
            print("✅ KẾT QUẢ TRÍCH XUẤT VĂN BẢN (TEXT EXTRACTED):")
            print("="*50)
            print(result["text"])
            print("="*50)
        else:
            print("⚠️ Không tìm thấy chữ nào trong ảnh.")
            
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {str(e)}")

if __name__ == "__main__":
    main()
