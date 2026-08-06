from __future__ import annotations


def analyze_transactions(
    transactions: list[dict[str, int | str]],
    limits: dict[str, int],
    retry_count: int,
    mode: str,
) -> str:
    """
    İşlem kayıtlarını, kategori limitlerini, yeniden deneme
    sayısını ve çalışma modunu birlikte değerlendirir.

    Bu fonksiyon; iç içe döngüler, break/continue ifadeleri,
    sözlük ve liste erişimleri, yerel sayaçlar, exception akışı,
    bileşik koşullar ve erken return ifadeleri içerir.
    """

    if retry_count < 0:
        return "Geçersiz deneme sayısı"

    if retry_count >= 5:
        return "İşlem sistemi kilitli"

    if mode not in {
        "STRICT",
        "NORMAL",
        "RELAXED",
    }:
        return "Geçersiz çalışma modu"

    if not transactions:
        return "İşlem bulunamadı"

    if not limits:
        return "Limit bilgisi bulunamadı"

    valid_transaction_count = 0
    rejected_transaction_count = 0
    total_amount = 0
    suspicious_category_count = 0

    category_totals: dict[str, int] = {}

    for transaction in transactions:
        if not isinstance(transaction, dict):
            rejected_transaction_count += 1
            continue

        try:
            category = transaction["category"]
            amount = transaction["amount"]
        except KeyError:
            rejected_transaction_count += 1
            continue

        if not isinstance(category, str):
            rejected_transaction_count += 1
            continue

        if not isinstance(amount, int):
            rejected_transaction_count += 1
            continue

        if amount < 0:
            return "Negatif işlem tutarı"

        if category == "":
            rejected_transaction_count += 1
            continue

        if amount == 0:
            continue

        category_limit = limits.get(
            category,
        )

        if category_limit is None:
            suspicious_category_count += 1

            if mode == "STRICT":
                return "Tanımsız kategori"

            if mode == "NORMAL":
                rejected_transaction_count += 1
                continue

            category_limit = 1000

        if category_limit < 0:
            return "Geçersiz kategori limiti"

        category_total = category_totals.get(
            category,
            0,
        )

        remaining_attempts = retry_count + 1

        while remaining_attempts > 0:
            if amount <= category_limit:
                break

            amount -= 10
            remaining_attempts -= 1

        if amount > category_limit:
            rejected_transaction_count += 1
            continue

        category_total += amount
        category_totals[category] = category_total

        total_amount += amount
        valid_transaction_count += 1

        if category_total > category_limit * 2:
            suspicious_category_count += 1

        if (
            mode == "STRICT"
            and suspicious_category_count >= 2
        ):
            return "Şüpheli işlem yoğunluğu"

    if valid_transaction_count == 0:
        if rejected_transaction_count > 0:
            return "Tüm işlemler reddedildi"

        return "Geçerli işlem bulunamadı"

    try:
        average_amount = (
            total_amount / valid_transaction_count
        )
    except ZeroDivisionError:
        return "Ortalama hesaplanamadı"

    category_count = len(category_totals)

    if category_count == 0:
        return "Kategori özeti oluşturulamadı"

    if (
        mode == "STRICT"
        and rejected_transaction_count > valid_transaction_count
    ):
        return "Reddedilen işlem oranı yüksek"

    if (
        mode == "NORMAL"
        and suspicious_category_count >= 3
    ):
        return "Manuel inceleme gerekli"

    if (
        mode == "RELAXED"
        and average_amount >= 500
    ):
        return "Yüksek hacimli işlem"

    if total_amount > 10000:
        return "Toplam tutar limiti aşıldı"

    if average_amount >= 1000:
        return (
            "Yüksek ortalamalı işlem: "
            f"{average_amount:.2f}"
        )

    if rejected_transaction_count == 0:
        return (
            "Tüm işlemler kabul edildi: "
            f"{valid_transaction_count}"
        )

    return (
        "İşlem analizi tamamlandı: "
        f"{valid_transaction_count} kabul, "
        f"{rejected_transaction_count} red"
    )


def calculate_category_usage(
    current_amount: int,
    category_limit: int,
) -> float:
    """Kategori limitinin kullanım yüzdesini hesaplar."""
    if current_amount < 0:
        return 0.0

    if category_limit <= 0:
        return 0.0

    return round(
        current_amount / category_limit * 100,
        2,
    )


def determine_transaction_risk(
    amount: int,
    retry_count: int,
    is_unknown_category: bool,
) -> str:
    """İşlem için temel risk seviyesi oluşturur."""
    if amount < 0 or retry_count < 0:
        return "INVALID"

    if is_unknown_category and amount >= 1000:
        return "CRITICAL"

    if retry_count >= 4 or amount >= 5000:
        return "HIGH"

    if retry_count >= 2 or amount >= 1000:
        return "MEDIUM"

    return "LOW"