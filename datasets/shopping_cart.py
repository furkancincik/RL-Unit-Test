class Product:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock

    def update_stock(self, amount):
        if self.stock + amount < 0:
            raise ValueError("Stock cannot be negative.")

        self.stock += amount


class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add_product(self, product, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")

        if product.stock < quantity:
            return False

        if product.name in self.items:
            self.items[product.name]["quantity"] += quantity
        else:
            self.items[product.name] = {
                "product": product,
                "quantity": quantity
            }

        product.update_stock(-quantity)
        return True

    def remove_product(self, product_name):
        if product_name not in self.items:
            return False

        item = self.items[product_name]
        product = item["product"]
        quantity = item["quantity"]

        product.update_stock(quantity)
        del self.items[product_name]

        return True

    def calculate_total(self, customer_type="normal", coupon=None):
        total = 0

        for item in self.items.values():
            product = item["product"]
            quantity = item["quantity"]

            total += product.price * quantity

        if customer_type == "premium":
            total *= 0.90
        elif customer_type == "student":
            total *= 0.95

        if coupon == "SAVE20" and total >= 200:
            total -= 20
        elif coupon == "SAVE50" and total >= 500:
            total -= 50

        return round(total, 2)

    def get_most_expensive_item(self):
        if not self.items:
            return None

        most_expensive = None

        for item in self.items.values():
            product = item["product"]

            if most_expensive is None:
                most_expensive = product
            elif product.price > most_expensive.price:
                most_expensive = product

        return most_expensive.name

    def summary(self):
        result = []

        for name, item in self.items.items():
            result.append({
                "name": name,
                "quantity": item["quantity"],
                "unit_price": item["product"].price
            })

        return result


def process_order(cart, customer_type, coupon=None):
    if not cart.items:
        return {
            "success": False,
            "message": "Cart is empty."
        }

    try:
        total = cart.calculate_total(customer_type, coupon)

        if total <= 0:
            return {
                "success": False,
                "message": "Invalid order total."
            }

        if total >= 1000:
            shipping = 0
        elif total >= 300:
            shipping = 10
        else:
            shipping = 25

        final_total = total + shipping

        return {
            "success": True,
            "subtotal": total,
            "shipping": shipping,
            "total": final_total,
            "most_expensive_item": cart.get_most_expensive_item()
        }

    except (ValueError, TypeError) as error:
        return {
            "success": False,
            "message": str(error)
        }


if __name__ == "__main__":
    laptop = Product("Laptop", 750, 5)
    mouse = Product("Mouse", 40, 10)
    keyboard = Product("Keyboard", 80, 4)

    cart = ShoppingCart()

    cart.add_product(laptop, 1)
    cart.add_product(mouse, 2)
    cart.add_product(keyboard, 1)

    result = process_order(
        cart,
        customer_type="premium",
        coupon="SAVE50"
    )

    print("Cart:")
    for item in cart.summary():
        print(item)

    print("\nOrder result:")
    print(result)