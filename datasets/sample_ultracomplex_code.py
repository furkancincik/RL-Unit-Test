from __future__ import annotations


def process_order(
    amount: float,
    stock: int,
    customer_type: str,
    coupon: str,
    failed_attempts: int,
    items: list[int],
) -> str:
    """
    Sipariş tutarı, stok, müşteri türü, kupon, başarısız deneme
    sayısı ve ürün listesini birlikte değerlendirir.

    Bu fonksiyon; bileşik koşullar, döngüler, exception akışları,
    erken return ifadeleri, dinamik return ve bilinçli olarak
    erişilemeyen bir dal içerir.
    """
    if amount < 0:
        return "Geçersiz tutar"

    if stock < 0:
        return "Geçersiz stok"

    if failed_attempts < 0:
        return "Geçersiz deneme sayısı"

    if failed_attempts >= 5:
        return "Hesap kilitli"

    if not items:
        return "Sepet boş"

    try:
        first_item = items[0]
    except IndexError:
        return "Sepet erişim hatası"

    if first_item < 0:
        return "Geçersiz ürün değeri"

    valid_item_count = 0
    total_item_value = 0

    for item in items:
        if item < 0:
            continue

        total_item_value += item
        valid_item_count += 1

    if valid_item_count == 0:
        return "Geçerli ürün yok"

    try:
        average_item_value = (
            total_item_value / valid_item_count
        )
    except ZeroDivisionError:
        return "Ortalama hesaplanamadı"

    discount = 0

    if customer_type == "VIP":
        discount += 20
    elif customer_type == "MEMBER":
        discount += 10
    elif customer_type != "GUEST":
        return "Geçersiz müşteri türü"

    if coupon == "SAVE10" and amount >= 100:
        discount += 10
    elif coupon == "SAVE5" or amount >= 500:
        discount += 5
    elif coupon not in ("NONE", ""):
        return "Geçersiz kupon"

    remaining_checks = 2

    while remaining_checks > 0:
        remaining_checks -= 1

    if stock == 0:
        return "Stok yok"

    if stock < valid_item_count:
        return "Yetersiz stok"

    if amount == 0:
        return "Ücretsiz sipariş"

    if amount > 10000:
        return "Manuel inceleme"

    if (
        customer_type == "VIP"
        and discount >= 25
        and average_item_value >= 50
    ):
        return "VIP özel sipariş"

    if discount >= 20 and amount >= 200:
        return "Yüksek indirimli sipariş"

    if average_item_value >= 100:
        return (
            "Yüksek değerli sipariş: "
            f"{average_item_value:.2f}"
        )

    if amount >= 100:
        return (
            "Onaylandı: "
            f"{amount - (amount * discount / 100):.2f}"
        )

    if amount < 100:
        return "Minimum tutar sağlanmadı"

    # Bilinçli olarak erişilemeyen dal:
    # amount >= 100 ve amount < 100 kontrolleri tüm olasılıkları
    # tükettiği için buraya ulaşılamaz.
    if amount == 50:
        return "Erişilemeyen özel durum"

    return "Sipariş reddedildi"


def calculate_shipping_cost(
    amount: float,
    distance: int,
) -> float:
    """Sipariş tutarı ve mesafeye göre kargo maliyeti hesaplar."""
    if amount < 0 or distance < 0:
        return 0.0

    if amount >= 500:
        return 0.0

    if distance <= 10:
        return 20.0

    return round(
        20.0 + (distance - 10) * 1.5,
        2,
    )


def calculate_fraud_risk(
    amount: float,
    failed_attempts: int,
) -> str:
    """Sipariş için basit dolandırıcılık risk seviyesi üretir."""
    if amount < 0 or failed_attempts < 0:
        return "INVALID"

    if failed_attempts >= 5 or amount > 10000:
        return "CRITICAL"

    if failed_attempts >= 3 and amount >= 1000:
        return "HIGH"

    if failed_attempts >= 1 or amount >= 500:
        return "MEDIUM"

    return "LOW"