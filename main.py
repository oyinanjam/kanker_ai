from src.train_model import train_random_forest
from src.predict import predict_risk

def input_data_pasien():
    # input dari user
    print("Masukkan data pasien untuk prediksi kekambuhan kanker:")
    radius_mean = float(input("Radius Mean: "))
    texture_mean = float(input("Texture Mean: "))
    perimeter_mean = float(input("Perimeter Mean: "))
    area_mean = float(input("Area Mean: "))
    smoothness_mean = float(input("Smoothness Mean: "))
    compactness_mean = float(input("Compactness Mean: "))
    concavity_mean = float(input("Concavity Mean: "))
    concave_points_mean = float(input("Concave Points Mean: "))
    symmetry_mean = float(input("Symmetry Mean: "))
    fractal_dimension_mean = float(input("Fractal Dimension Mean: "))
    radius_se = float(input("Radius SE: "))
    texture_se = float(input("Texture SE: "))
    perimeter_se = float(input("Perimeter SE: "))
    area_se = float(input("Area SE: "))
    smoothness_se = float(input("Smoothness SE: "))
    compactness_se = float(input("Compactness SE: "))
    concavity_se = float(input("Concavity SE: "))
    concave_points_se = float(input("Concave Points SE: "))
    symmetry_se = float(input("Symmetry SE: "))
    fractal_dimension_se = float(input("Fractal Dimension SE: "))
    radius_worst = float(input("Radius Worst: "))
    texture_worst = float(input("Texture Worst: "))
    perimeter_worst = float(input("Perimeter Worst: "))
    area_worst = float(input("Area Worst: "))
    smoothness_worst = float(input("Smoothness Worst: "))
    compactness_worst = float(input("Compactness Worst: "))
    concavity_worst = float(input("Concavity Worst: "))
    concave_points_worst = float(input("Concave Points Worst: "))
    symmetry_worst = float(input("Symmetry Worst: "))
    fractal_dimension_worst = float(input("Fractal Dimension Worst: "))

    return {
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'smoothness_mean': [smoothness_mean],
        'compactness_mean': [compactness_mean],
        'concavity_mean': [concavity_mean],
        'concave_points_mean': [concave_points_mean],
        'symmetry_mean': [symmetry_mean],
        'fractal_dimension_mean': [fractal_dimension_mean],
        'radius_se': [radius_se],
        'texture_se': [texture_se],
        'perimeter_se': [perimeter_se],
        'area_se': [area_se],
        'smoothness_se': [smoothness_se],
        'compactness_se': [compactness_se],
        'concavity_se': [concavity_se],
        'concave_points_se': [concave_points_se],
        'symmetry_se': [symmetry_se],
        'fractal_dimension_se': [fractal_dimension_se],
        'radius_worst': [radius_worst],
        'texture_worst': [texture_worst],
        'perimeter_worst': [perimeter_worst],
        'area_worst': [area_worst],
        'smoothness_worst': [smoothness_worst],
        'compactness_worst': [compactness_worst],
        'concavity_worst': [concavity_worst],
        'concave_points_worst': [concave_points_worst],
        'symmetry_worst': [symmetry_worst],
        'fractal_dimension_worst': [fractal_dimension_worst]
    }

if __name__ == "__main__":
    data_path = "data/cancer_data.csv"
    model_path = "model/kanker_model.pkl"

    print("=== Melatih Model ===")
    train_random_forest(data_path, model_path)

    print("\n=== Memprediksi Risiko Kekambuhan ===")
    
    input_data_user = input_data_pasien()

    prediction = predict_risk(input_data_user, model_path)
    
    print("Prediksi: ", "Kekambuhan" if prediction[0] == 1 else "Tidak Ada Kekambuhan")
