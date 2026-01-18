from src.inference import predict_image


class AppController:
    def __init__(self, view, model_path):
        self.view = view
        self.model_path = model_path

    def analyze(self, image_path):
        result = predict_image(self.model_path, image_path)

        #GUI günceller
        self.view.update_traits(result["traits"])
        self.view.update_confidence(
            result["model_confidence"],
            result["entropy_confidence"]
        )
