#!/usr/bin/env python3
"""
行將估價系統 - Flask API Server
啟動方式：python api_server.py

Endpoints:
  GET  /health              - 健康檢查
  POST /predict/auto        - 汽車估價
  POST /predict/moto        - 機車估價
  POST /predict/batch/auto  - 汽車批量估價（多台）
"""

import os
import sys

# 確保 scripts/ 在路徑中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.train_model import AutomobilePredictor, MotorcyclePredictor

app = Flask(__name__)
CORS(app)  # 允許跨來源請求（UI 與 API 分開部署時必備）

# 全域 predictor 實例（程式啟動時只載入一次）
auto_model_path = os.path.join(os.path.dirname(__file__), "data", "models", "automobile_v1.pkl")
moto_model_path = os.path.join(os.path.dirname(__file__), "data", "models", "motorcycle_v1.pkl")

ap = AutomobilePredictor(auto_model_path)
mp = MotorcyclePredictor(moto_model_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_auto": auto_model_path, "model_moto": moto_model_path})


@app.route("/predict/auto", methods=["POST"])
def predict_auto():
    """
    汽車估價

    Body (JSON):
    {
      "brand": "BENZ",
      "model": "C200",
      "year": 2019,
      "month": 6,
      "cc": 1497,
      "grade": "B+",
      "mileage_km": 89208,
      "mileage_available": 1,
      "transmission": "手自",
      "tax_total": 0,
      "auction_date": "2026-06-19"
    }

    Response:
    {
      "predicted_price": 651250,
      "price_lower": 346000,
      "price_upper": 960000,
      "confidence": "high",
      "mileage_unavailable": false,
      "note": ""
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        result = ap.predict(data)
        r = result
        return jsonify({
            "predicted_price": r.get("price_estimate", r.get("predicted_price")),
            "price_lower": r.get("lower", r.get("price_lower")),
            "price_upper": r.get("upper", r.get("price_upper")),
            "confidence": "high" if data.get("mileage_available", 1) == 1 else "medium",
            "mileage_unavailable": data.get("mileage_available", 1) == 0,
            "note": ""
        })

    except FileNotFoundError as e:
        return jsonify({"error": f"模型檔案找不到：{e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/moto", methods=["POST"])
def predict_moto():
    """
    機車估價

    Body (JSON):
    {
      "brand": "YAMAHA",
      "model": "GRYPHUS",
      "year": 2022,
      "month": 1,
      "cc": 155,
      "grade": "A",
      "mileage_km": 12000,
      "mileage_available": 1,
      "auction_date": "2026-06-19"
    }

    Response:
    {
      "predicted_price": 39000,
      "price_lower": 10000,
      "price_upper": 521000,
      "confidence": "high",
      "mileage_unavailable": false,
      "note": ""
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        result = mp.predict(data)
        r = result
        return jsonify({
            "predicted_price": r.get("price_estimate", r.get("predicted_price")),
            "price_lower": r.get("lower", r.get("price_lower")),
            "price_upper": r.get("upper", r.get("price_upper")),
            "confidence": "high" if data.get("mileage_available", 1) == 1 else "medium",
            "mileage_unavailable": data.get("mileage_available", 1) == 0,
            "note": ""
        })

    except FileNotFoundError as e:
        return jsonify({"error": f"模型檔案找不到：{e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch/auto", methods=["POST"])
def predict_batch_auto():
    """
    汽車批量估價（一次估多台）

    Body (JSON):
    {
      "vehicles": [
        {"brand": "BENZ", "model": "C200", "year": 2019, ...},
        {"brand": "BMW", "model": "320i", "year": 2020, ...}
      ]
    }
    """
    try:
        data = request.get_json()
        vehicles = data.get("vehicles", [])
        results = []
        for v in vehicles:
            try:
                results.append({"input": v, "result": ap.predict(v)})
            except Exception as e:
                results.append({"input": v, "error": str(e)})
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="行將估價系統 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="監聽位址（預設 0.0.0.0）")
    parser.add_argument("--port", type=int, default=5050, help="監聽連接埠（預設 5050）")
    parser.add_argument("--debug", action="store_true", help="啟用 Flask debug 模式")
    args = parser.parse_args()

    print(f"🚀 啟動行將估價系統 API Server")
    print(f"   汽車模型：{auto_model_path}")
    print(f"   機車模型：{moto_model_path}")
    print(f"   監聽：http://{args.host}:{args.port}")
    print()
    print("可用端點：")
    print("  GET  /health")
    print("  POST /predict/auto")
    print("  POST /predict/moto")
    print("  POST /predict/batch/auto")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
