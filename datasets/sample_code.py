def calculate_score(score: int) -> str:
    if score >= 85:
        return "Başarılı"

    if score >= 50:
        return "Orta"

    return "Başarısız"
