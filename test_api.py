import unittest
import requests

class TestAPI(unittest.TestCase):
    def test_prediction(self):
        # Отправляем POST-запрос с изображением в теле запроса
        with open('image.jpg', 'rb') as f:
            response = requests.post('http://ramilsaf.ru/predict', files={'file': f})

        # Проверяем, что ответ имеет код 200
        self.assertEqual(response.status_code, 200)

        # Проверяем, что ответ содержит ожидаемый текст
        self.assertIn('Результат', response.text)

if __name__ == '__main__':
    unittest.main()
