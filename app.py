
from flask import *
import Crypto
from Crypto.PublicKey import RSA
from Crypto import Random
import ast




app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def basic():
    if request.method=="POST":
        req_data=request.get_json()
        print(req_data['recording'])
    return jsonify({'messege':'Success'})   

if __name__=='__main__':
    app.run(debug=True)


