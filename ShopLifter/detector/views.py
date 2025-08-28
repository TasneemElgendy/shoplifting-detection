# detector/views.py 
import os, tempfile 
from django.http import JsonResponse, HttpResponse 
from django.views.decorators.csrf import csrf_exempt 
from django.shortcuts import render 
from .inference import predict_from_video_file 

# Map numeric labels to human-readable text 
CLASS_LABELS = { 
    0: "Non-shoplifter", 
    1: "Shoplifter" 
} 

def upload_page(request): 
    return render(request, "upload.html") 

def home(request): 
    return HttpResponse("✅ Model loaded. Use POST /predict with form-data field 'video'.") 

@csrf_exempt  # for local testing only 
def predict(request): 
    if request.method != "POST": 
        return JsonResponse({"error": "Use POST with form-data field 'video'."}, status=405) 

    if "video" not in request.FILES: 
        return JsonResponse({"error": "Missing 'video' file."}, status=400) 

    uploaded = request.FILES["video"] 
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4" 
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp: 
        for chunk in uploaded.chunks(): 
            tmp.write(chunk) 
        tmp_path = tmp.name 

    try: 
        print("📂 Received video:", uploaded.name, "-> saved to:", tmp_path)

        # Call inference
        raw_result = predict_from_video_file(tmp_path) 
        print("🔍 RAW_RESULT >>>", raw_result, type(raw_result))

        # Ensure result is always dict
        if not isinstance(raw_result, dict): 
            print("⚠️ RAW_RESULT was not dict, wrapping it.")
            raw_result = {"label": raw_result}

        label_val = raw_result.get("label") 
        print("🎯 label_val before mapping:", label_val, type(label_val))

        # Map numeric → human-readable
        if isinstance(label_val, (int, float)) or (isinstance(label_val, str) and label_val.isdigit()): 
            label_val = CLASS_LABELS.get(int(label_val), str(label_val)) 

        print("✅ Final label_val:", label_val)

        # Final result
        result = {**raw_result, "label": label_val} 
        print("📦 Final result dict:", result)

        return JsonResponse({
            "ok": True,
            "label": result["label"],
            "score": result.get("score"),
            "id": result.get("id")
        })

    except Exception as e: 
        print("🔥 Exception occurred:", repr(e)) 
        return JsonResponse({"ok": False, "error": str(e)}, status=500) 
    finally: 
        try: 
            os.remove(tmp_path) 
            print("🗑️ Deleted temp file:", tmp_path)
        except Exception as cleanup_err: 
            print("⚠️ Failed to delete temp file:", tmp_path, "Error:", repr(cleanup_err))
