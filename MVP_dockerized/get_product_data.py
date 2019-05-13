from flask import Flask, request
from predict_service import RecommendationHandler

app = Flask(__name__)

@app.route('/recommend', methods = ['GET'])

def get_product():
	product_category = request.args['product_category']
	video_category = request.args['video_category']

	print(product_category)
	print(video_category)

	rh = RecommendationHandler()
	recommendations = rh.recommend_channel(product_category, video_category)

	return recommendations

if __name__ == "__main__":
	app.run(debug=True, port=3000)