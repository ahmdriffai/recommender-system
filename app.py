import pickle
from flask import Flask, jsonify
app = Flask(__name__)
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import requests

productapi = requests.get('http://127.0.0.1:8000/api/product').json()
product = pd.DataFrame(productapi)
product.rename(columns={
    'id': 'product_id'
}, inplace=True)

ratingapi = requests.get('http://127.0.0.1:8000/api/rating').json()
rating = pd.DataFrame(ratingapi)

df = pd.merge(rating, product, on='product_id', how='inner')

utility = df.pivot(index = 'name', columns = 'user_id', values = 'rating')
utility = utility.fillna(0)


a = rating.groupby('product_id')
b = a.first()
b['id'].index[1]

# similarity = 1- distance
# distance_mtx = squareform(pdist(utility, 'cosine'))
# similarity_mtx = 1- distance_mtx

item_similarity = utility.T.corr()
similarity_mtx = item_similarity.to_numpy()

def calculate_user_rating(userid, similarity_mtx, utility):
    user_rating = utility.iloc[:,userid-1]
    pred_rating = deepcopy(user_rating)
    
    default_rating = user_rating[user_rating>0].mean()
    numerate = np.dot(similarity_mtx, user_rating)
    corr_sim = similarity_mtx[:, user_rating >0]
    for i,ix in enumerate(pred_rating):
        temp = 0
        if ix < 1:
            w_r = numerate[i]
            sum_w = corr_sim[i,:].sum()
            if w_r == 0 or sum_w == 0:
                temp = default_rating
            else:
                temp = w_r / sum_w
            pred_rating.iloc[i] = temp
    return pred_rating


def recommendation_to_user(userid, top_n, similarity_mtx, utility):
    user_rating = utility.iloc[:,userid-1]
    pred_rating = calculate_user_rating(userid, similarity_mtx, utility)

    top_item = sorted(range(1,len(pred_rating)), key = lambda i: -1*pred_rating.iloc[i])
    top_item = list(filter(lambda x: user_rating.iloc[x]==0, top_item))[:top_n]
    res = []
    for i in top_item:
        res.append({"id": b['id'].index[i].item(), "pred" : pred_rating.iloc[i]})

    return res

product = pd.read_csv('data/products.csv')

@app.route('/predict/<int:userid>/<int:top_n>')
def index(userid, top_n):
    predict = recommendation_to_user(userid, top_n , similarity_mtx, utility)
    responses = []
    for i in predict:
        image = product.loc[product['id'] == i['id']]['image'].values[0]
        name = product.loc[product['id'] == i['id']]['name'].values[0]

        responses.append({'id': i['id'], "name": name, "image": image})

    return jsonify(responses)


if __name__ == '__main__':
    app.run(debug = True)
    app.run(host='0.0.0.0', port=105)

# predict = recommendation_to_user(1, 3 , similarity_mtx, utility)

# for i in predict:
#     print(i['id'])
# print(predict)
# responses = []
# for i in predict:
#     image = product.loc[product['id'] == i['id']]['image'].values[0]
#     name = product.loc[product['id'] == i['id']]['name'].values[0]

#     responses.append({'id': i['id'], "name": name, "image": image})

# print(responses)

# name = product.loc[product['id'] == 2]['name'].to_string(index=False)

# predict = recommendation_to_user(2, 30 , similarity_mtx, utility)
# responses = []
# for i in predict:
#     image = product.loc[product['id'] == i['id']]['image'].values[0]
#     name = product.loc[product['id'] == i['id']]['name'].values[0]

#     responses.append({'id': i['id'], "name": name, "image": image})

# print(responses)
