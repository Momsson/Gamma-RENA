# Radiation Dose Calculation

def calculate_dose(exposure_time, dose_rate):
    """
    Calculate the radiation dose received over a period of time.
    
    :param exposure_time: time of exposure in hours
    :param dose_rate: dose rate in mSv/h
    :return: total dose in mSv
    """
    return exposure_time * dose_rate


def health_risk_assessment(dose):
    """
    Assess health risks based on radiation dose.
    
    :param dose: total dose in mSv
    :return: risk assessment
    """
    if dose < 1:
        return "Minimal risk"
    elif dose < 10:
        return "Low risk"
    elif dose < 50:
        return "Moderate risk"
    else:
        return "High risk"


# Example usage
if __name__ == '__main__':
    exposure_time = 5  # hours
    dose_rate = 2  # mSv/h
    total_dose = calculate_dose(exposure_time, dose_rate)
    risk = health_risk_assessment(total_dose)
    print(f'Total dose: {total_dose} mSv')
    print(f'Health risk: {risk}')