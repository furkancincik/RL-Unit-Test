from __future__ import annotations


def evaluate_application(
    score: int,
    attendance: int,
    project_score: int,
    disciplinary_points: int,
) -> str:
    """
    Öğrencinin puan, devam, proje ve disiplin durumunu birlikte
    değerlendirerek ayrıntılı bir başvuru sonucu üretir.

    Bu fonksiyon; çok sayıda parametre, iç içe karar yapıları,
    sınır değerleri ve erken return ifadeleri içerdiği için
    otomatik test üretimi ve RL coverage optimizasyonunu daha
    kapsamlı biçimde doğrulamak amacıyla hazırlanmıştır.
    """

    if score < 0:
        return "Geçersiz sınav puanı"

    if score > 100:
        return "Geçersiz sınav puanı"

    if attendance < 0:
        return "Geçersiz devam oranı"

    if attendance > 100:
        return "Geçersiz devam oranı"

    if project_score < 0:
        return "Geçersiz proje puanı"

    if project_score > 100:
        return "Geçersiz proje puanı"

    if disciplinary_points < 0:
        return "Geçersiz disiplin puanı"

    if disciplinary_points > 100:
        return "Geçersiz disiplin puanı"

    if disciplinary_points >= 80:
        return "Disiplin nedeniyle reddedildi"

    if attendance < 40:
        return "Devamsızlık nedeniyle reddedildi"

    if score >= 90:
        if attendance >= 90:
            if project_score >= 90:
                if disciplinary_points == 0:
                    return "Tam burs"

                if disciplinary_points <= 10:
                    return "Üstün başarı bursu"

                return "Yüksek başarı"

            if project_score >= 70:
                return "Akademik başarı"

            return "Proje geliştirmesi gerekli"

        if attendance >= 70:
            if project_score >= 80:
                return "Başarılı başvuru"

            return "Koşullu akademik kabul"

        return "Devam oranı yetersiz"

    if score >= 70:
        if attendance >= 80:
            if project_score >= 85:
                if disciplinary_points <= 20:
                    return "Başarı bursu"

                return "Disiplin incelemesi"

            if project_score >= 60:
                return "Normal kabul"

            return "Proje puanı yetersiz"

        if attendance >= 60:
            if project_score >= 75:
                return "Koşullu kabul"

            return "Ek değerlendirme gerekli"

        return "Devam desteği gerekli"

    if score >= 50:
        if attendance >= 75:
            if project_score >= 80:
                return "Proje başarısıyla kabul"

            if project_score >= 50:
                return "Sınırda kabul"

            return "Proje nedeniyle reddedildi"

        if attendance >= 50:
            if disciplinary_points <= 15:
                return "Mülakat gerekli"

            return "Disiplin ve başarı incelemesi"

        return "Devamsızlık nedeniyle başarısız"

    if score == 0:
        if attendance == 0:
            return "Başvuruya katılmadı"

        return "Sınava katılmadı"

    if project_score >= 90:
        if attendance >= 85:
            return "Özel yetenek değerlendirmesi"

        return "Proje başarılı ancak devam yetersiz"

    if disciplinary_points >= 50:
        return "Başarısız ve disiplinli"

    return "Başvuru reddedildi"


def calculate_risk_level(
    disciplinary_points: int,
) -> str:
    """Disiplin puanına göre risk seviyesi hesaplar."""
    if disciplinary_points < 0:
        return "INVALID"

    if disciplinary_points >= 80:
        return "CRITICAL"

    if disciplinary_points >= 50:
        return "HIGH"

    if disciplinary_points >= 20:
        return "MEDIUM"

    return "LOW"


def determine_interview_priority(
    score: int,
    project_score: int,
) -> str:
    """Başvuru için mülakat önceliği belirler."""
    if score < 0:
        return "INVALID"

    if project_score < 0:
        return "INVALID"

    if score >= 85:
        if project_score >= 85:
            return "FIRST"

        return "SECOND"

    if score >= 60:
        if project_score >= 75:
            return "SECOND"

        return "THIRD"

    return "NONE"