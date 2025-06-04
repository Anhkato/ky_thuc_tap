from book_management import PhysicalBook, EBook
from library_management import User, Library, display_books

def main():
    print("Chào mừng đến với Chương trình Quản lý Thư viện Sách Trực Tuyến!")

    # 1. Tạo các đối tượng sách và thêm vào thư viện
    print("\n--- Khởi tạo Thư viện và Sách ---")
    try:
        book1 = PhysicalBook("PB001", "Lập trình Python từ A đến Z", "Nguyễn Văn A", 10, "Mới")
        book2 = PhysicalBook("PB002", "Cấu trúc dữ liệu và Giải thuật", "Trần Thị B", 5, "Đã qua sử dụng")
        book3 = PhysicalBook("PB003", "Toán rời rạc ứng dụng", "Lê Văn C", 0, "Mới")
        ebook1 = EBook("EB001", "Machine Learning Cơ Bản", "Phạm Văn D", 50, "PDF")
        ebook2 = EBook("EB002", "Deep Learning Nâng Cao", "Hoàng Thị E", 100, "EPUB")
    except ValueError as e:
        print(f"Lỗi khởi tạo sách: {e}")
        return

    my_library = Library()
    my_library.add_book(book1)
    my_library.add_book(book2)
    my_library.add_book(book3)
    my_library.add_book(ebook1)
    my_library.add_book(ebook2)
    my_library.add_book(PhysicalBook("PB001", "Sách trùng lặp", "Tác giả X", 5, "Cũ"))


    # 2. Tạo người dùng
    print("\n--- Tạo Người dùng ---")
    try:
        user1 = User("USR001", "Nguyễn Văn An")
        user2 = User("USR002", "Trần Thị Bình")
        print(user1)
        print(user2)
    except ValueError as e:
        print(f"Lỗi tạo người dùng: {e}")
        return

    # 3. Thực hiện mượn sách
    print("\n--- Giao dịch Mượn Sách ---")
    def attempt_borrow(user: User, book_id: str, library: Library):
        print(f"\n{user.name} thử mượn sách mã '{book_id}':")
        target_book = library.find_book(book_id)
        if target_book:
            if target_book.stock > 0:
                if target_book.update_stock(-1):
                    if user.borrow_book(book_id):
                        print(f"Giao dịch thành công: {user.name} mượn '{target_book.title}'. Kho còn: {target_book.stock}.")
                    else:
                        target_book.update_stock(1) 
                        print(f"Lỗi: {user.name} đã mượn '{target_book.title}' rồi.")
                else:
                    print(f"Lỗi: Không đủ sách '{target_book.title}' trong kho.")
            else:
                print(f"Lỗi: Sách '{target_book.title}' (mã: {book_id}) đã hết hàng.")
        else:
            print(f"Lỗi: Không tìm thấy sách có mã '{book_id}'.")

    attempt_borrow(user1, "PB001", my_library)
    attempt_borrow(user1, "EB001", my_library)
    attempt_borrow(user1, "PB003", my_library)
    attempt_borrow(user1, "XX001", my_library)
    attempt_borrow(user1, "PB001", my_library)

    attempt_borrow(user2, "PB002", my_library)
    attempt_borrow(user2, "EB002", my_library)

    # 4. Thực hiện trả sách
    print("\n--- Giao dịch Trả Sách ---")
    def attempt_return(user: User, book_id: str, library: Library):
        print(f"\n{user.name} thử trả sách mã '{book_id}':")
        target_book = library.find_book(book_id)
        if user.return_book(book_id): 
            if target_book:
                target_book.update_stock(1)
                print(f"Giao dịch thành công: {user.name} trả '{target_book.title}'. Kho hiện có: {target_book.stock}.")
            else:
                print(f"Giao dịch thành công: {user.name} trả sách mã '{book_id}' (sách không còn trong thư viện).")
        else:
            print(f"Lỗi: {user.name} không mượn sách có mã '{book_id}'.")


    attempt_return(user1, "PB001", my_library)
    attempt_return(user1, "NNN00", my_library)


    # 5. Sử dụng Iterator để duyệt và in thông tin tất cả sách trong thư viện
    print("\n--- Danh sách sách trong thư viện (Iterator, sắp xếp theo tiêu đề) ---")
    try:
        for book_item in my_library:
            print(book_item.get_info())
    except Exception as e:
        print(f"Lỗi khi duyệt thư viện: {e}")
    print("-----------------------------------------------------------------------------")

    # 6. Gọi hàm display_books để hiển thị thông tin sách (thể hiện tính đa hình)
    print("\n--- Danh sách chi tiết các sách (Đa hình) ---")
    display_books(my_library.books)
    print("--------------------------------------------")


    # 7. In danh sách sách đang mượn của từng người dùng
    print("\n--- Danh sách sách đang mượn của người dùng ---")
    print(f"{user1.name} đang mượn:")
    borrowed_ids_user1 = user1.get_borrowed_books()
    if borrowed_ids_user1:
        for book_id_usr1 in borrowed_ids_user1:
            book_usr1 = my_library.find_book(book_id_usr1)
            if book_usr1: print(f" - {book_usr1.get_info()}")
            else: print(f" - Mã sách: {book_id_usr1} (Không có trong thư viện)")
    else:
        print(" (Không mượn sách nào)")

    print(f"\n{user2.name} đang mượn:")
    borrowed_ids_user2 = user2.get_borrowed_books()
    if borrowed_ids_user2:
        for book_id_usr2 in borrowed_ids_user2:
            book_usr2 = my_library.find_book(book_id_usr2)
            if book_usr2: print(f" - {book_usr2.get_info()}")
            else: print(f" - Mã sách: {book_id_usr2} (Không có trong thư viện)")
    else:
        print(" (Không mượn sách nào)")
    print("----------------------------------------------")

    print("\nChương trình Quản lý Thư viện kết thúc.")

if __name__ == "__main__":
    main()