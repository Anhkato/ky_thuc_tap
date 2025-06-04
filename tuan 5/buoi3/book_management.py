class Book:
    def __init__(self, book_id: str, title: str, author: str, stock: int):
        if not book_id:
            raise ValueError("Mã sách không được để trống.")
        if stock < 0:
            raise ValueError("Số lượng tồn kho không thể âm.")
        self.__book_id = book_id
        self.title = title
        self.author = author
        self.stock = stock

    def get_book_id(self) -> str:
        return self.__book_id

    def get_info(self) -> str:
        return f"Mã sách: {self.__book_id}, Tiêu đề: {self.title}, Tác giả: {self.author}, Tồn kho: {self.stock}"

    def update_stock(self, quantity_change: int) -> bool:
        if self.stock + quantity_change >= 0:
            self.stock += quantity_change
            return True
        else:
            print(f"Lỗi: Không đủ sách '{self.title}' trong kho để thực hiện giao dịch.")

    def __str__(self) -> str:
        return self.get_info()

class PhysicalBook(Book):
    def __init__(self, book_id: str, title: str, author: str, stock: int, physical_status: str):
        super().__init__(book_id, title, author, stock)
        self.physical_status = physical_status

    def get_info(self) -> str:
        return f"{super().get_info()}, Trạng thái: {self.physical_status}"

class EBook(Book):
    def __init__(self, book_id: str, title: str, author: str, stock: int, file_format: str):
        super().__init__(book_id, title, author, stock)
        self.file_format = file_format

    def get_info(self) -> str:
        return f"{super().get_info()}, Định dạng: {self.file_format}"