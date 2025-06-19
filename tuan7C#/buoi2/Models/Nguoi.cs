namespace HeThongQuanLyHocVien.Models
{
    public abstract class Nguoi
    {
        private string _ten;
        private string _ho;
        private string _email;

        public string Ho
        {
            get { return _ho; }
            private set { _ho = value; }
        }

        public string Ten
        {
            get { return _ten; }
            private set { _ten = value; }
        }

        public string HoTen => $"{Ho} {Ten}";

        public string Email
        {
            get { return _email; }
            protected set
            {
                if (string.IsNullOrWhiteSpace(value) || !value.Contains("@"))
                {
                    throw new ArgumentException("Email không hợp lệ.");
                }
                _email = value;
            }
        }

        public Nguoi(string ho, string ten, string email)
        {
            Ho = ho;
            Ten = ten;
            Email = email;
        }

        public virtual void HienThiThongTin()
        {
            Console.WriteLine($"Họ Tên: {HoTen}, Email: {Email}");
        }
    }
}
