using System.Diagnostics;

namespace SolAI.Pipecat.LLMService.Infrastructure;

public static class ToolExecutionLogger
{
    public static string Execute(string pluginName, string functionName, Func<string> action)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            var result = action();
            stopwatch.Stop();
            Log(pluginName, functionName, result, stopwatch.Elapsed, isError: false);
            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            Log(pluginName, functionName, ex.Message, stopwatch.Elapsed, isError: true);
            throw;
        }
    }

    private static void Log(string pluginName, string functionName, string payload, TimeSpan elapsed, bool isError)
    {
        var previousColor = Console.ForegroundColor;
        var status = isError ? "error" : "ok";
        var elapsedMs = elapsed.TotalMilliseconds.ToString("0.##");

        try
        {
            Console.ForegroundColor = isError ? ConsoleColor.Red : ConsoleColor.DarkYellow;
            Console.WriteLine();
            Console.WriteLine($"[tool] {pluginName}.{functionName} [{status}] [{elapsedMs} ms] -> {payload}");
        }
        finally
        {
            Console.ForegroundColor = previousColor;
        }
    }
}
