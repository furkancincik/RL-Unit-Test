from __future__ import annotations


def calculate_score(score: int) -> str:
    """
    Sayısal puanı ayrıntılı kurallara göre sınıflandırır.

    Farklı koşullar, iç içe karar noktaları ve sınır değerleri
    sayesinde CFG, DQM, senaryo üretimi ve RL akışının daha
    kapsamlı biçimde doğrulanmasını sağlar.
    """
    if score < 0:
        return "Geçersiz"

    if score > 100:
        return "Geçersiz"

    if score >= 90:
        if score == 100:
            return "Mükemmel"

        return "Çok Başarılı"

    if score >= 75:
        if score >= 85:
            return "Başarılı"

        return "İyi"

    if score >= 50:
        if score >= 60:
            return "Orta"

        return "Geçer"

    if score == 0:
        return "Katılmadı"

    return "Başarısız"


def calculate_letter_grade(score: int) -> str:
    """Puanı harf notuna dönüştürür."""
    if score < 0 or score > 100:
        return "INVALID"

    if score >= 90:
        return "AA"

    if score >= 85:
        return "BA"

    if score >= 75:
        return "BB"

    if score >= 65:
        return "CB"

    if score >= 55:
        return "CC"

    if score >= 50:
        return "DC"

    return "FF"


def calculate_bonus(
    score: int,
    attendance: int,
) -> int:
    """
    Puan ve devam oranına göre bonus hesaplar.
    """
    if score < 0 or attendance < 0:
        return 0

    bonus = 0

    if attendance >= 90:
        bonus += 5
    elif attendance >= 75:
        bonus += 2

    if score >= 85:
        bonus += 5
    elif score >= 70:
        bonus += 3

    return bonus


def calculate_average(
    scores: list[int],
) -> float:
    """
    Geçerli puanların ortalamasını hesaplar.
    """
    if not scores:
        return 0.0

    total = 0
    valid_count = 0

    for score in scores:
        if score < 0 or score > 100:
            continue

        total += score
        valid_count += 1

    if valid_count == 0:
        return 0.0

    return round(
        total / valid_count,
        2,
    )


def evaluate_student(
    score: int,
    attendance: int,
) -> str:
    """
    Puan ve devam durumunu birlikte değerlendirir.
    """
    if score < 0 or score > 100:
        return "Geçersiz puan"

    if attendance < 0 or attendance > 100:
        return "Geçersiz devam oranı"

    if attendance < 50:
        return "Devamsızlıktan başarısız"

    if score >= 85 and attendance >= 85:
        return "Üstün başarı"

    if score >= 50:
        return "Başarılı"

    return "Başarısız"