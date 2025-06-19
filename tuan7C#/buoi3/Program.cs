using LibraryManagement.Models;
using LibraryManagement.Services;
using System;

namespace LibraryManagement
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            LibraryManager manager = new LibraryManager();

            while (true)
            {
                Console.WriteLine("\n================ HỆ THỐNG QUẢN LÝ THƯ VIỆN ================");
                Console.WriteLine("1.  Hiển thị toàn bộ sách trong kho");
                Console.WriteLine("2.  Thêm sách mới vào kho");
                Console.WriteLine("3.  Tìm kiếm sách theo thể loại hoặc tác giả");
                Console.WriteLine("4.  Xử lý mượn sách");
                Console.WriteLine("5.  Xử lý trả sách");
                Console.WriteLine("6.  Thống kê sách theo thể loại");
                Console.WriteLine("7.  Hiển thị Top 3 sách có số lượng nhiều nhất");
                Console.WriteLine("8.  Hiển thị danh sách người đang mượn sách");
                Console.WriteLine("9.  Hiển thị lịch sử mượn/trả");
                Console.WriteLine("10. Hiển thị danh sách các thể loại");
                Console.WriteLine("11. Hiển thị hàng chờ mượn sách");
                Console.WriteLine("0.  Thoát chương trình");
                Console.Write("Vui lòng chọn một chức năng: ");

                string choice = Console.ReadLine();

                switch (choice)
                {
                    case "1":
                        manager.DisplayAllBooks();
                        break;
                    case "2":
                        AddNewBook(manager);
                        break;
                    case "3":
                        SearchBook(manager);
                        break;
                    case "4":
                        BorrowBook(manager);
                        break;
                    case "5":
                        ReturnBook(manager);
                        break;
                    case "6":
                        manager.GroupByGenre();
                        break;
                    case "7":
                        manager.DisplayTop3MostBooks();
                        break;
                    case "8":
                        manager.DisplayWhoBorrowsWhat();
                        break;
                    case "9":
                        manager.DisplayTransactionHistory();
                        break;
                    case "10":
                        manager.DisplayAllGenres();
                        break;
                    case "11":
                        manager.DisplayWaitingQueue();
                        break;
                    case "0":
                        Console.WriteLine("Cảm ơn đã sử dụng hệ thống!");
                        return;
                    default:
                        Console.WriteLine("Lựa chọn không hợp lệ. Vui lòng chọn lại.");
                        break;
                }
                Console.WriteLine("\nNhấn phím bất kỳ để tiếp tục...");
                Console.ReadKey();
                Console.Clear();
            }
        }

        static void AddNewBook(LibraryManager manager)
        {
            try
            {
                Console.Write("Nhập ID sách: ");
                int id = int.Parse(Console.ReadLine());
                Console.Write("Nhập tiêu đề sách: ");
                string title = Console.ReadLine();
                Console.Write("Nhập tác giả: ");
                string author = Console.ReadLine();
                Console.Write("Nhập thể loại: ");
                string genre = Console.ReadLine();
                Console.Write("Nhập số lượng: ");
                int quantity = int.Parse(Console.ReadLine());

                manager.AddBook(new Book(id, title, author, genre, quantity));
                Console.WriteLine("Thêm/Cập nhật sách thành công!");
            }
            catch (FormatException)
            {
                Console.WriteLine("Lỗi: ID và số lượng phải là số.");
            }
        }

       static void SearchBook(LibraryManager manager)
            {
                Console.WriteLine("\n--- Tìm kiếm sách ---");

                var authors = manager.GetAllAuthors();
                var genres = manager.GetAllGenres().ToList(); 

                Console.WriteLine("Gợi ý: Các tác giả hiện có:");
                Console.WriteLine($"   {string.Join(", ", authors)}");

                Console.WriteLine("\nGợi ý: Các thể loại hiện có:");
                Console.WriteLine($"   {string.Join(", ", genres)}\n");


                Console.Write("Nhập tên tác giả hoặc thể loại cần tìm: ");
                string keyword = Console.ReadLine();
                var results = manager.SearchByGenreOrAuthor(keyword);

                if (results.Any())
                {
                    Console.WriteLine($"\n--- Kết quả tìm kiếm cho '{keyword}' ---");
                    foreach (var book in results)
                    {
                        Console.WriteLine(book);
                    }
                }
                else
                {
                    Console.WriteLine("Không tìm thấy sách nào phù hợp.");
                }
            }

        static void BorrowBook(LibraryManager manager)
        {
            try
            {
                Console.Write("Nhập tên của bạn: ");
                string borrowerName = Console.ReadLine();
                Console.Write("Nhập ID sách bạn muốn mượn: ");
                int bookId = int.Parse(Console.ReadLine());
                manager.ProcessBorrowRequest(borrowerName, bookId);
            }
            catch (FormatException)
            {
                Console.WriteLine("Lỗi: ID sách phải là một con số.");
            }
        }

        static void ReturnBook(LibraryManager manager)
        {
            try
            {
                Console.Write("Nhập tên của bạn: ");
                string borrowerName = Console.ReadLine();
                Console.Write("Nhập ID sách bạn muốn trả: ");
                int bookId = int.Parse(Console.ReadLine());
                manager.ProcessReturnRequest(borrowerName, bookId);
            }
            catch (FormatException)
            {
                Console.WriteLine("Lỗi: ID sách phải là một con số.");
            }
        }
    }
}