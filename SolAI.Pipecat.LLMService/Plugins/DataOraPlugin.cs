using System.ComponentModel;
using SolAI.Pipecat.LLMService.Infrastructure;
using Microsoft.SemanticKernel;

namespace SolAI.Pipecat.LLMService.Plugins;

public sealed class DataOraPlugin
{
 [KernelFunction("ottieni_data_ora_corrente")]
 [Description("Restituisce la data e l'ora correnti del sistema in formato locale.")]
 public string OttieniDataOraCorrente()
 {
  return ToolExecutionLogger.Execute("DataOra", "ottieni_data_ora_corrente", () =>
  {
   return DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
  });
 }

 [KernelFunction("ottieni_data_corrente")]
 [Description("Restituisce la data corrente del sistema in formato locale.")]
 public string OttieniDataCorrente()
 {
  return ToolExecutionLogger.Execute("DataOra", "ottieni_data_corrente", () =>
  {
   return DateTime.Now.ToString("yyyy-MM-dd");
  });
 }

 [KernelFunction("ottieni_ora_corrente")]
 [Description("Restituisce l'ora corrente del sistema in formato locale.")]
 public string OttieniOraCorrente()
 {
  return ToolExecutionLogger.Execute("DataOra", "ottieni_ora_corrente", () =>
  {
   return DateTime.Now.ToString("HH:mm:ss");
  });
 }
}
