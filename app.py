from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved models
with open("model-pkh.pkl", "rb") as f:
    model_pkh = pickle.load(f)

with open("model-bpnt.pkl", "rb") as f:
    model_bpnt = pickle.load(f)

# Ordinal mappings for categorical features
ordinal_mapping_pkh = {
   'Usia': {'<25': 0.25, '≥25 - <35': 0.5, '≥35 - ≤50': 0.75, '>50': 1},
    'Status': {'Belum Kawin': 0.33, 'Sudah Kawin': 0.66, 'Cerai': 1},
    'Penghasilan': {'0-500.000': 1, '500.000 - 2.000.000': 0.75, '2.000.000 - 5.000.000': 0.5, '>5.000.000': 0.25},
    'Kepemilikan_Rumah': {'Bebas Sewa': 1, 'Kontrak': 0.75, 'Milik Orang Tua': 0.5, 'Milik Sendiri': 0.25},
    'Luas_Lantai': {'≤ 8 m2': 1, '≥ 8 m2': 0.5},
    'Jenis_Lantai': {'Tanah': 1, 'Semen': 0.75, 'Keramik': 0.5, 'Granit': 0.25},
    'Jenis_Dinding': {'Bambu': 1, 'Kayu': 0.66, 'Tembok': 0.33},
    'Jenis_Atap': {'Jerami': 1, 'Seng': 0.66, 'Genteng': 0.33},
    'Sumber_Air': {'Sungai': 1, 'Sumur': 0.75, 'Ledeng': 0.5, 'PDAM': 0.25},
    'Jaringan_Listrik': {'Listrik PLN': 0.5, 'Non Listrik PLN': 1},
    'Bhn_msk': {'Kayu': 1, 'Minyak Tanah': 0.75, 'Gas LPG': 0.5, 'Listrik': 0.25},
    'Kepemilikan_Asset': {'Tanah': 0.25, 'Mobil': 0.5, 'Motor': 0.75, 'Tidak Punya': 1},
}

ordinal_mapping_bpnt = {
   'Usia': {'<25': 0.25, '≥25 - <35': 0.5, '≥35 - ≤50': 0.75, '>50': 1},
    'Status': {'Belum Kawin': 0.33, 'Sudah Kawin': 0.66, 'Cerai': 1},
    'Pendidikan': {'SD': 1, 'SMP': 0.8, 'SLTA': 0.6, 'S1': 0.4, 'S2': 0.2},
    'Pekerjaan': {'Belum/Tidak Bekerja': 1, 'Buruh': 0.75, 'Wiraswasta': 0.5, 'Swasta': 0.25},
    'Penghasilan': {'0-500.000': 1, '500.000 - 2.000.000': 0.75, '2.000.000 - 5.000.000': 0.5, '>5.000.000': 0.25},
    'Tanggungan': {'>4': 1, '4': 0.8, '3': 0.6, '2': 0.4, '1': 0.2},
}

# Define a route for the index
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Define a route for the prediction using model-pkh
@app.route('/predict_pkh', methods=['POST'])
def predict_pkh():
    try:
             # Get the JSON data from the raw request
        data = request.get_json()

        # Extracting values from the JSON data
        NIK = data.get('NIK')
        Usia_str = data.get('Usia')
        Status_str = data.get('Status')
        Penghasilan = data.get('Penghasilan')
        Kepemilikan_Rumah = data.get('Kepemilikan_Rumah')
        Luas_Lantai = data.get('Luas_Lantai')
        Jenis_Lantai = data.get('Jenis_Lantai')
        Jenis_Dinding = data.get('Jenis_Dinding')
        Jenis_Atap = data.get('Jenis_Atap')
        Sumber_Air = data.get('Sumber_Air')
        Jaringan_Listrik = data.get('Jaringan_Listrik')
        Bhn_msk = data.get('Bhn_msk')
        Kepemilikan_Asset = data.get('Kepemilikan_Asset')

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[NIK, Usia_str, Status_str, Penghasilan, Kepemilikan_Rumah, Luas_Lantai,
                                     Jenis_Lantai, Jenis_Dinding, Jenis_Atap, Sumber_Air, Jaringan_Listrik,
                                     Bhn_msk, Kepemilikan_Asset]],
                                   columns=['NIK', 'Usia', 'Status', 'Penghasilan', 'Kepemilikan_Rumah', 'Luas_Lantai',
                                            'Jenis_Lantai', 'Jenis_Dinding', 'Jenis_Atap', 'Sumber_Air', 'Jaringan_Listrik',
                                            'Bhn_msk', 'Kepemilikan_Asset'])



        # Convert to the ordinal value using the mapping
        input_data['Usia'] = input_data['Usia'].map(ordinal_mapping_pkh['Usia'])
        input_data['Status'] = input_data['Status'].map(ordinal_mapping_pkh['Status'])
        input_data['Penghasilan'] = input_data['Penghasilan'].map(ordinal_mapping_pkh['Penghasilan'])
        input_data['Kepemilikan_Rumah'] = input_data['Kepemilikan_Rumah'].map(ordinal_mapping_pkh['Kepemilikan_Rumah'])
        input_data['Luas_Lantai'] = input_data['Luas_Lantai'].map(ordinal_mapping_pkh['Luas_Lantai'])
        input_data['Jenis_Lantai'] = input_data['Jenis_Lantai'].map(ordinal_mapping_pkh['Jenis_Lantai'])
        input_data['Jenis_Dinding'] = input_data['Jenis_Dinding'].map(ordinal_mapping_pkh['Jenis_Dinding'])
        input_data['Jenis_Atap'] = input_data['Jenis_Atap'].map(ordinal_mapping_pkh['Jenis_Atap'])
        input_data['Sumber_Air'] = input_data['Sumber_Air'].map(ordinal_mapping_pkh['Sumber_Air'])
        input_data['Jaringan_Listrik'] = input_data['Jaringan_Listrik'].map(ordinal_mapping_pkh['Jaringan_Listrik'])
        input_data['Bhn_msk'] = input_data['Bhn_msk'].map(ordinal_mapping_pkh['Bhn_msk'])
        input_data['Kepemilikan_Asset'] = input_data['Kepemilikan_Asset'].map(ordinal_mapping_pkh['Kepemilikan_Asset'])

        # Use the model to make a prediction
        output = model_pkh.predict(input_data)

        # Set the threshold for eligibility
        threshold = 0.7

        # Determine eligibility status and amount based on the ranking
        eligibility_status = "Layak" if threshold <= output < 0.85 else "Layak" if output >= 0.85 else "Tidak Layak"
        dana_amount = "Rp.500.000" if threshold <= output < 0.85 else "Rp.1.000.000" if output >= 0.85 else "Rp.0"

        # Construct the response dictionary
        response_dict = {
            'NIK': NIK,
            'Status': eligibility_status,
            'Dana': dana_amount
        }

        # Return JSON response
        return jsonify(response_dict)

    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route for the prediction using model-bpnt
@app.route('/predict_bpnt', methods=['POST'])
def predict_bpnt():
    try:
       # Get the JSON data from the raw request
        data = request.get_json()

        # Extracting values from the JSON data
        NIK = data.get('NIK')
        Usia_str = data.get('Usia')
        Status_str = data.get('Status')
        Pendidikan = data.get('Pendidikan')
        Pekerjaan = data.get('Pekerjaan')
        Penghasilan = data.get('Penghasilan')
        Tanggungan_str = data.get('Tanggungan')

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[NIK, Usia_str, Status_str, Pendidikan, Pekerjaan, Penghasilan, Tanggungan_str]],
                                   columns=['NIK', 'Usia', 'Status', 'Pendidikan', 'Pekerjaan', 'Penghasilan', 'Tanggungan'])

        # Convert to the ordinal value using the mapping
        input_data['Usia'] = input_data['Usia'].map(ordinal_mapping_bpnt['Usia'])
        input_data['Status'] = input_data['Status'].map(ordinal_mapping_bpnt['Status'])
        input_data['Pendidikan'] = input_data['Pendidikan'].map(ordinal_mapping_bpnt['Pendidikan'])
        input_data['Pekerjaan'] = input_data['Pekerjaan'].map(ordinal_mapping_bpnt['Pekerjaan'])
        input_data['Penghasilan'] = input_data['Penghasilan'].map(ordinal_mapping_bpnt['Penghasilan'])
        input_data['Tanggungan'] = input_data['Tanggungan'].map(ordinal_mapping_bpnt['Tanggungan'])

        # Use the model to make a prediction
        output = model_bpnt.predict(input_data)

        # Set the threshold for eligibility
        threshold = 0.8

        # Determine the eligibility status based on the ranking
        eligibility_status = "Layak" if output >= threshold else "Tidak Layak"
        dana_amount = "Rp.600.000 atau 60kg Beras" if output >= threshold else "Rp.0"

        # Construct the response dictionary
        response_dict = {
            'NIK': NIK,
            'Status': eligibility_status,
            'Dana': dana_amount
        }

        # Return JSON response
        return jsonify(response_dict)

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
	app.run(debug=True)


