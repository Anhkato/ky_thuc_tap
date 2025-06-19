using LibraryManagement.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace LibraryManagement.Services
{
    public class LibraryManager
    {
        private List<Book> books;
        private Stack<string> transactionHistory;
        private Dictionary<string, List<int>> borrowedBooksByUser;
        private Queue<Tuple<string, int>> waitingQueue; 
        private HashSet<string> genres;

        public LibraryManager()
        {
            books = new List<Book>();
            transactionHistory = new Stack<string>();
            borrowedBooksByUser = new Dictionary<string, List<int>>();
            waitingQueue = new Queue<Tuple<string, int>>();
            genres = new HashSet<string>();
            
            // Thêm một vài dữ liệu mẫu
            InitializeData();
        }

        private void InitializeData()
        {
            AddBook(new Book(1, "Lão Hạc", "Nam Cao", "Văn học", 1));
            AddBook(new Book(2, "Số Đỏ", "Vũ Trọng Phụng", "Văn học", 5));
            AddBook(new Book(3, "Dế Mèn Phiêu Lưu Ký", "Tô Hoài", "Thiếu nhi", 15));
            AddBook(new Book(4, "Clean Code", "Robert C. Martin", "Lập trình", 7));
            AddBook(new Book(5, "The Pragmatic Programmer", "Andrew Hunt", "Lập trình", 3));
        }

        public void AddBook(Book newBook)
        {
            var existingBook = books.FirstOrDefault(b => b.ID == newBook.ID);
            if (existingBook != null)
            {
                existingBook.Quantity += newBook.Quantity;
            }
            else
            {
                books.Add(newBook);
            }
            genres.Add(newBook.Genre);
        }

        public List<Book> SearchByGenreOrAuthor(string keyword)
        {
            return books.Where(b => b.Genre.Equals(keyword, StringComparison.OrdinalIgnoreCase) ||
                                    b.Author.Equals(keyword, StringComparison.OrdinalIgnoreCase))
                        .ToList();
        }

        public void GroupByGenre()
        {
            Console.WriteLine("\n--- Thống kê sách theo thể loại ---");
            var groupedBooks = books.GroupBy(b => b.Genre);
            foreach (var group in groupedBooks)
            {
                Console.WriteLine($"Thể loại: {group.Key} - Tổng số cuốn: {group.Sum(b => b.Quantity)}");
            }
        }

        public bool ProcessBorrowRequest(string borrowerName, int bookId)
        {
            var book = books.FirstOrDefault(b => b.ID == bookId);
            if (book == null)
            {
                Console.WriteLine("Lỗi: Không tìm thấy sách với ID này.");
                return false;
            }

            if (book.Quantity > 0)
            {
                book.Quantity--;
                transactionHistory.Push($"Mượn: Sách ID {bookId} bởi '{borrowerName}' lúc {DateTime.Now}");

                if (!borrowedBooksByUser.ContainsKey(borrowerName))
                {
                    borrowedBooksByUser[borrowerName] = new List<int>();
                }
                borrowedBooksByUser[borrowerName].Add(bookId);

                Console.WriteLine($"'{borrowerName}' đã mượn thành công sách '{book.Title}'.");
                return true;
            }
            else
            {
                waitingQueue.Enqueue(new Tuple<string, int>(borrowerName, bookId));
                Console.WriteLine($"Sách '{book.Title}' đã hết. Yêu cầu của bạn đã được thêm vào hàng chờ.");
                return false;
            }
        }

        public void ProcessReturnRequest(string borrowerName, int bookId)
        {
            var book = books.FirstOrDefault(b => b.ID == bookId);
            if (book == null)
            {
                Console.WriteLine("Lỗi: Không tìm thấy sách với ID này.");
                return;
            }

            if (!borrowedBooksByUser.ContainsKey(borrowerName) || !borrowedBooksByUser[borrowerName].Contains(bookId))
            {
                Console.WriteLine($"Lỗi: Người dùng '{borrowerName}' không có mượn sách với ID {bookId}.");
                return;
            }

            book.Quantity++;
            borrowedBooksByUser[borrowerName].Remove(bookId);
            if (borrowedBooksByUser[borrowerName].Count == 0)
            {
                borrowedBooksByUser.Remove(borrowerName);
            }
            transactionHistory.Push($"Trả: Sách ID {bookId} bởi '{borrowerName}' lúc {DateTime.Now}");
            Console.WriteLine($"Sách '{book.Title}' đã được trả thành công.");

            CheckWaitingQueue(bookId);
        }
        
        private void CheckWaitingQueue(int bookId)
        {
            if (waitingQueue.Any(req => req.Item2 == bookId))
            {
                var waitingRequest = waitingQueue.FirstOrDefault(req => req.Item2 == bookId);
                var newQueue = new Queue<Tuple<string, int>>(waitingQueue.Where(req => req != waitingRequest));
                waitingQueue = newQueue;

                Console.WriteLine($"\nThông báo: Sách ID {bookId} đã có sẵn. Tự động xử lý mượn cho '{waitingRequest.Item1}' từ hàng chờ.");
                ProcessBorrowRequest(waitingRequest.Item1, waitingRequest.Item2);
            }
        }

        public void DisplayTop3MostBooks()
        {
            Console.WriteLine("\n--- Top 3 sách có số lượng nhiều nhất ---");
            var topBooks = books.OrderByDescending(b => b.Quantity).Take(3);
            foreach (var book in topBooks)
            {
                Console.WriteLine(book);
            }
        }

        public void DisplayWhoBorrowsWhat()
        {
            Console.WriteLine("\n--- Danh sách người dùng đang mượn sách ---");
            if (borrowedBooksByUser.Count == 0)
            {
                Console.WriteLine("Chưa có ai mượn sách.");
                return;
            }

            var borrowedDetails = borrowedBooksByUser
                .SelectMany(user => user.Value.Select(bookId => new { UserName = user.Key, BookId = bookId }))
                .Join(books,
                      borrowInfo => borrowInfo.BookId,
                      book => book.ID,
                      (borrowInfo, book) => $"{borrowInfo.UserName} đang mượn: '{book.Title}' (ID: {book.ID})");

            foreach (var detail in borrowedDetails)
            {
                Console.WriteLine(detail);
            }
        }
        public List<string> GetAllAuthors()
        {
            return books
            .Select(b => b.Author)
            .Distinct()
            .ToList();
        }  
        public List<string> GetAllGenres()
        {
            return genres.ToList();
        }    


        public void DisplayAllBooks()
        {
            Console.WriteLine("\n--- Toàn bộ sách trong kho ---");
            foreach (var book in books)
            {
                Console.WriteLine(book);
            }
        }

        public void DisplayTransactionHistory()
        {
            Console.WriteLine("\n--- Lịch sử Mượn/Trả ---");
            foreach (var entry in transactionHistory)
            {
                Console.WriteLine(entry);
            }
        }
        
        public void DisplayWaitingQueue()
        {
            Console.WriteLine("\n--- Hàng chờ mượn sách ---");
            if (waitingQueue.Count == 0)
            {
                Console.WriteLine("Hàng chờ trống.");
                return;
            }
            foreach (var request in waitingQueue)
            {
                Console.WriteLine($"Người chờ: {request.Item1}, Sách ID: {request.Item2}");
            }
        }

        public void DisplayAllGenres()
        {
            Console.WriteLine("\n--- Danh sách các thể loại sách ---");
            foreach (var genre in genres)
            {
                Console.WriteLine($"- {genre}");
            }
        }
    }
}