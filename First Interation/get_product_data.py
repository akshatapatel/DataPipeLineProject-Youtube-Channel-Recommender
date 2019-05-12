from flask import Flask, request
from predict_service import RecommendationHandler

app = Flask(__name__)

@app.route('/recommend', methods = ['GET'])

def get_product():
	product_category = request.args['product_category']
	video_category = request.args['video_category']
	product_description = request.args['product_description']
	weightage = request.args['feature_weights']


	# product_category = 'fashion and beauty'
	# video_category = 'style and beauty blogs'
	# product_description = 'this is an eyeshadow palette'
	# weightage = [0.3,0.3,0.4]



	rh = RecommendationHandler()
	recommendations = rh.recommend_channel(product_category, video_category, product_description, weightage)

	return recommendations

if __name__ == "__main__":
	app.run(debug=True, port=3000)