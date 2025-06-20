using System.Diagnostics;

class Program
{
    private static CancellationTokenSource? _cts;

    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        bool running = true;
        while (running)
        {
            Console.WriteLine("\n===== HỆ THỐNG PHÂN TÍCH TIN TỨC =====");
            Console.WriteLine("1. Bắt đầu tải tin tức đồng thời");
            Console.WriteLine("2. So sánh hiệu suất Thread và Task");
            Console.WriteLine("3. Thoát");
            Console.Write("Chọn một chức năng: ");

            string? choice = Console.ReadLine();
            switch (choice)
            {
                case "1":
                    await FetchAllNewsAsync();
                    break;
                case "2":
                    CompareThreadVsTask();
                    break;
                case "3":
                    running = false;
                    break;
                default:
                    Console.WriteLine("Lựa chọn không hợp lệ. Vui lòng chọn lại.");
                    break;
            }
        }
        Console.WriteLine("Chương trình đã thoát.");
    }

    static async Task FetchAllNewsAsync()
    {
        _cts = new CancellationTokenSource();
        var downloader = new NewsDownloader();
        var sources = new List<string> { "VNExpress", "Tuổi Trẻ", "Thanh Niên", "BBC", "CNN", "Reuters" };
        
        Console.WriteLine("\n*** Bắt đầu quá trình tải tin. Nhấn 'Q' bất kỳ lúc nào để hủy. ***");

        var cancellationListener = Task.Run(() =>
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                if (Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Q)
                {
                    Console.WriteLine("\n[HỦY] Đã nhận yêu cầu hủy! Đang dừng các tác vụ...");
                    _cts.Cancel();
                    break;
                }
                Task.Delay(100).Wait();
            }
        });

        var stopwatch = Stopwatch.StartNew();
        Console.WriteLine("\n--- Sử dụng Parallel.ForEachAsync ---");
        var successfulResults = new System.Collections.Concurrent.ConcurrentBag<string>();

        try
        {
            await Parallel.ForEachAsync(sources, _cts.Token, async (source, token) =>
            {
                try
                {
                    string result = await downloader.GetNewsAsync(source, token);
                    successfulResults.Add(result);
                }
                catch (HttpRequestException ex)
                {
                    Console.WriteLine($"[LỖI] Lỗi khi tải từ {source}: {ex.Message}");
                }
            });
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("[HỦY] Quá trình Parallel.ForEachAsync đã được hủy bỏ.");
        }
        
        stopwatch.Stop();
        Console.WriteLine("\n--- KẾT QUẢ TỔNG HỢP ---");
        Console.WriteLine($"Tổng thời gian tải: {stopwatch.Elapsed.TotalSeconds:F2} giây.");
        
        if (successfulResults.Any())
        {
            long totalChars = successfulResults.Sum(r => r.Length);
            Console.WriteLine($"Tải thành công từ {successfulResults.Count}/{sources.Count} nguồn.");
            Console.WriteLine($"Tổng số ký tự từ các nguồn thành công: {totalChars}.");
        }
        else
        {
            Console.WriteLine("Không có nguồn nào được tải thành công.");
        }

        if (!cancellationListener.IsCompleted)
        {
            _cts.Cancel(); 
            await Task.WhenAny(cancellationListener, Task.Delay(500));
        }
    }

    static void CompareThreadVsTask()
    {
        Console.WriteLine("\n--- So sánh hiệu suất Thread vs. Task ---");
        const int iterations = 3;
        const int delay = 1000;

        var stopwatch = Stopwatch.StartNew();
        var threads = new List<Thread>();
        for (int i = 0; i < iterations; i++)
        {
            var t = new Thread(() =>
            {
                Console.WriteLine($"Thread {Thread.CurrentThread.ManagedThreadId} bắt đầu và ngủ.");
                Thread.Sleep(delay);
            });
            threads.Add(t);
            t.Start();
        }
        foreach (var t in threads)
        {
            t.Join();
        }
        stopwatch.Stop();
        Console.WriteLine($"Tổng thời gian của {iterations} Thread chạy song song: {stopwatch.Elapsed.TotalSeconds:F2} giây.");

        stopwatch.Restart();
        var tasks = new List<Task>();
        for (int i = 0; i < iterations; i++)
        {
            tasks.Add(Task.Run(async () =>
            {
                Console.WriteLine($"Task trên thread {Thread.CurrentThread.ManagedThreadId} bắt đầu và delay.");
                await Task.Delay(delay);
            }));
        }
        Task.WhenAll(tasks).Wait();
        stopwatch.Stop();
        Console.WriteLine($"Tổng thời gian của {iterations} Task chạy song song: {stopwatch.Elapsed.TotalSeconds:F2} giây.");
        Console.WriteLine("Ghi chú: Cả hai đều chạy gần như song song. Tuy nhiên, Task hiệu quả hơn về tài nguyên do sử dụng Thread Pool, đặc biệt tốt cho các tác vụ chờ (I/O-bound).");
    }
}