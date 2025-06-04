from typing import List, Iterator
from book_management import Book

class User:
    def __init__(self, user_id: str, name: str):
        if not user_id:
            raise ValueError("Mã người dùng không được để trống.")
        self.__user_id = user_id
        self.name = name
        self.borrowed_books: List[str] = []

    def get_user_id(self) -> str:
        return self.__user_id

    def borrow_book(self, book_id: str) -> bool:
        if not book_id:
            print("Lỗi: Mã sách không hợp lệ.")
            return False
        if book_id not in self.borrowed_books:
            self.borrowed_books.append(book_id)
            print(f"Người dùng '{self.name}' đã mượn sách có mã '{book_id}'.")
            return True
        else:
            print(f"Lỗi: Người dùng '{self.name}' đã mượn sách có mã '{book_id}' trước đó.")
            return False

    def return_book(self, book_id: str) -> bool:
        if book_id in self.borrowed_books:
            self.borrowed_books.remove(book_id)
            print(f"Người dùng '{self.name}' đã trả sách có mã '{book_id}'.")
            return True
        else:
            print(f"Lỗi: Người dùng '{self.name}' không mượn sách có mã '{book_id}'.")
            return False

    def get_borrowed_books(self) -> List[str]:
        return self.borrowed_books

    def __str__(self) -> str:
        return f"Mã người dùng: {self.__user_id}, Tên: {self.name}, Sách đang mượn: {len(self.borrowed_books)} cuốn."


class Library:
    def __init__(self, books: List[Book] = None):
        self.books: List[Book] = books if books is not None else []
        self._current_index = 0

    def add_book(self, book: Book):
        if any(b.get_book_id() == book.get_book_id() for b in self.books):
            print(f"Lỗi: Sách với mã '{book.get_book_id()}' đã tồn tại trong thư viện.")
            return
        self.books.append(book)
        print(f"Đã thêm sách '{book.title}' vào thư viện.")

    def find_book(self, book_id: str) -> Book | None:
        for book in self.books:
            if book.get_book_id() == book_id:
                return book
        return None

    def __iter__(self) -> Iterator[Book]:
        self.books.sort(key=lambda book: book.title)
        self._current_index = 0
        return self

    def __next__(self) -> Book:
        if self._current_index < len(self.books):
            book = self.books[self._current_index]
            self._current_index += 1
            return book
        else:
            raise StopIteration

def display_books(book_list: List[Book]):
    if not book_list:
        print("Không có sách nào để hiển thị.")
        return
    print("\n--- Danh sách chi tiết các sách (Đa hình) ---")
    for book in book_list:
        print(book.get_info())
    print("--------------------------------------------")