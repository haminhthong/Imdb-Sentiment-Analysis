"""Kiểm thử tải (Load Test) tối giản cho API bằng thư viện chuẩn Python.

Cách sử dụng:
    Chạy API: python api.py
    Chạy load test: python load_test.py --users 50 --requests 200
"""

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def percentile(values: list[float], ratio: float) -> float:
    """Tính percentile theo nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(ratio * len(ordered)) - 1))
    return ordered[index]


def send_request(url: str, timeout: float) -> tuple[bool, float]:
    """Gửi một yêu cầu dự đoán và đo lường latency (ms)."""
    body = json.dumps({"text": "This movie has a wonderful story and great acting."}).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            success = response.status == 200
            response.read()
    except (HTTPError, URLError, TimeoutError):
        success = False
    return success, (time.perf_counter() - started) * 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm thử tải endpoint /predict")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.users <= 0 or args.requests <= 0 or args.timeout <= 0:
        raise ValueError("users, requests và timeout phải lớn hơn 0.")

    print(f"Đang chạy load test tới: {args.url}")
    print(f"Users đồng thời: {args.users} | Tổng số requests: {args.requests}")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.users) as executor:
        futures = [
            executor.submit(send_request, args.url, args.timeout) for _ in range(args.requests)
        ]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started

    latencies = [latency for success, latency in results if success]
    successes = len(latencies)
    print("=" * 50)
    print(f"Requests: {args.requests} | Thành công: {successes} | Lỗi: {args.requests - successes}")
    print(f"Thông lượng (Throughput): {successes / elapsed:.2f} req/s")
    if latencies:
        print(f"Mean Latency: {statistics.mean(latencies):.2f} ms")
        print(f"p50 Latency : {percentile(latencies, 0.50):.2f} ms")
        print(f"p95 Latency : {percentile(latencies, 0.95):.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
