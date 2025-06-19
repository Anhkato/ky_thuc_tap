namespace LibraryManagement.Models
{
    public class Book
    {
        public int ID { get; set; }
        public string Title { get; set; }
        public string Author { get; set; }
        public string Genre { get; set; }
        public int Quantity { get; set; }

        public Book(int id, string title, string author, string genre, int quantity)
        {
            ID = id;
            Title = title;
            Author = author;
            Genre = genre;
            Quantity = quantity;
        }

        public override string ToString()
        {
            return $"ID: {ID, -4} | Title: {Title, -30} | Author: {Author, -20} | Genre: {Genre, -15} | Quantity: {Quantity}";
        }
    }
}