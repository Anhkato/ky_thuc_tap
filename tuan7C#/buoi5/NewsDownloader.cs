using System.Net.Http;

public class NewsDownloader
{
    public async Task<string> GetNewsAsync(string source, CancellationToken token)
    {
        Console.WriteLine($"-> Bắt đầu tải từ {source}...");

        if (source == "CNN")
        {
            await Task.Delay(1000, token);
            throw new HttpRequestException($"Không thể kết nối đến máy chủ của {source}.");
        }
        
        await Task.Delay(new Random().Next(2000, 5001), token);

        Console.WriteLine($"<- Tải thành công từ {source}.");
        return $"Nội dung tin tức từ {source}.";
    }
}