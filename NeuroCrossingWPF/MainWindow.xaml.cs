using System.IO;
using System.Net;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using Accord.MachineLearning.Clustering;
using System.Drawing;
using ScottPlot;
using ScottPlot.WPF;
using Simple_NN;

namespace NeuroCrossingWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        static Random rnd = new Random();
        public  MainWindow()
        {
            InitializeComponent();

            Network network = new Network(new int[] { 2, 16, 16, 16, 16, 3 });
            Teacher teacher = new Teacher(0.001, 0.0005, network);
            network.LoadWeights("save.bin");

            //TestLearImage(network, teacher, 100000000);
            //return;

            // 1. Получаем данные для визуализации
            var points = GetPoints(network);
            List<(int layer, int neuron)> neurons = new List<(int, int)>();
            List<double[]> cords = new List<double[]>();

            foreach (var point in points)
            {
                neurons.Add(point.Key);
                cords.Add(point.Value.ToArray());
            }

            var pcords = cords.ToArray();

            // 2. Применяем t-SNE
            var tsne = new TSNE()
            {
                NumberOfInputs = 10,
                NumberOfOutputs = 2,
                Perplexity = 5
            };
            double[][] tsneResult = tsne.Transform(pcords);


            var dbscan = new DBSCAN(eps: 2.3, minPts: 1);
            int[] labels = dbscan.Fit(tsneResult);

            // 1. Создаём форму
            var wpfPlot = new WpfPlot();
            this.Content = wpfPlot;

            var plt = wpfPlot.Plot;

            // 4. Подготовка данных для отображения
            double[] xs = tsneResult.Select(x => x[0]).ToArray();
            double[] ys = tsneResult.Select(x => x[1]).ToArray();


            // 5. Создаем палитру цветов
            var colors = new ScottPlot.Color[]
{
    Colors.Gray,
Colors.Red,
Colors.Green,
Colors.Blue,
Colors.Orange,
Colors.Purple,
Colors.Cyan,
Colors.Magenta,
Colors.Coral,
Colors.Yellow,
Colors.Lime,
Colors.Indigo,
Colors.Teal,
Colors.Brown,
Colors.Pink,
Colors.Olive,
Colors.Black,
Colors.Maroon,
Colors.Navy,
Colors.Turquoise
};

            // 6. Добавляем точки с кластерами и готовим данные для изображений
            var clusterCenters = new Dictionary<int, (double x, double y)>();

            for (int cluster = labels.Min(); cluster < labels.Max() + 1; cluster++)
            {
                var indices = Enumerable.Range(0, labels.Length)
                                      .Where(i => labels[i] == cluster)
                                      .ToArray();

                var scatter = plt.Add.ScatterPoints(
                    indices.Select(i => xs[i]).ToArray(),
                    indices.Select(i => ys[i]).ToArray()
                );

                scatter.Color = colors[(cluster + 1) % colors.Length];
                scatter.MarkerSize = 15;
                scatter.MarkerShape = MarkerShape.Asterisk;
                scatter.Label = $"Кластер {cluster}";

                // Сохраняем центр кластера для позиционирования изображения
                if (indices.Length > 0)
                {
                    double centerX = indices.Select(i => xs[i]).Average();
                    double centerY = indices.Select(i => ys[i]).Average();
                    clusterCenters[cluster] = (centerX, centerY);
                }
            }

            wpfPlot.Refresh();
            this.Show();

            ClearDirectory("del");

            // 7. Создаем изображения с удаленными кластерами
            Bitmap img = new Bitmap("image.png");

            for (int clusForDel = -1; clusForDel < labels.Max() + 1; clusForDel++)
            {
                network.LoadWeights("save.bin");

                for (int i = 0; i < labels.Length; i++)
                {
                    if (labels[i] == clusForDel)
                    {
                        int l = neurons[i].layer;
                        int n = neurons[i].neuron;

                        for (int j = 0; j < network.neurons[l + 1].Length; j++)
                        {
                            network.weights[l][n, j] = 0.01;
                        }
                    }
                }

                Bitmap img2 = new Bitmap(img.Width, img.Height);
                for (int x = 0; x < img.Width; x++)
                {
                    for (int y = 0; y < img.Height; y++)
                    {
                        double[] C = network.Compute(new double[] { (double)x / img.Width, (double)y / img.Height });
                        System.Drawing.Color c = System.Drawing.Color.FromArgb((byte)(C[0] * 255D), (byte)(C[1] * 255D), (byte)(C[2] * 255D));
                        img2.SetPixel(x, y, c);
                    }
                }

                string imagePath = $"del\\image_{clusForDel}.png";
                img2.Save(imagePath);
                Console.WriteLine(clusForDel);

                // Добавляем маркер с изображением для каждого кластера (кроме шума -1)
                if (clusForDel >= 0 && clusterCenters.ContainsKey(clusForDel))
                {
                    var center = clusterCenters[clusForDel];
                    plt.Add.ImageMarker(
                        new Coordinates(center.x, center.y),
                        new ScottPlot.Image(imagePath),
                        0.15f
                    );
                    wpfPlot.Refresh();

                }
            }


            // 8. Сохраняем график с изображениями кластеров


            DirectoryCopy("del", $"backup\\{rnd.Next()}", true);
        }

        private static void DirectoryCopy(
        string sourceDirName, string destDirName, bool copySubDirs)
        {
            DirectoryInfo dir = new DirectoryInfo(sourceDirName);
            DirectoryInfo[] dirs = dir.GetDirectories();

            // If the source directory does not exist, throw an exception.
            if (!dir.Exists)
            {
                throw new DirectoryNotFoundException(
                    "Source directory does not exist or could not be found: "
                    + sourceDirName);
            }

            // If the destination directory does not exist, create it.
            if (!Directory.Exists(destDirName))
            {
                Directory.CreateDirectory(destDirName);
            }


            // Get the file contents of the directory to copy.
            FileInfo[] files = dir.GetFiles();

            foreach (FileInfo file in files)
            {
                // Create the path to the new copy of the file.
                string temppath = Path.Combine(destDirName, file.Name);

                // Copy the file.
                file.CopyTo(temppath, false);
            }

            // If copySubDirs is true, copy the subdirectories.
            if (copySubDirs)
            {

                foreach (DirectoryInfo subdir in dirs)
                {
                    // Create the subdirectory.
                    string temppath = Path.Combine(destDirName, subdir.Name);

                    // Copy the subdirectories.
                    DirectoryCopy(subdir.FullName, temppath, copySubDirs);
                }
            }
        }

        private static void ClearDirectory(string destDirName)
        {
            foreach (string file in Directory.GetFiles(destDirName))
            {
                File.Delete(file);
            }
        }

        static Dictionary<(int, int), List<double>> GetPoints(Network network)
        {
            var neuronFeatures = new Dictionary<(int, int), List<double>>();
            var outputs = new List<(double R, double G, double B)>();

            // 1. Собираем активации и выходы
            for (int i = 0; i < 100; i++)
            {
                // Генерируем случайный вход
                for (int j = 0; j < network.neurons[0].Length; j++)
                    network.neurons[0][j] = rnd.NextDouble();

                // Запускаем сеть
                network.Compute();
                outputs.Add((network.neurons.Last()[0], network.neurons.Last()[1], network.neurons.Last()[2]));

                // Записываем активации скрытых нейронов
                for (int l = 1; l < network.neurons.Length - 1; l++)
                {
                    for (int n = 0; n < network.neurons[l].Length; n++)
                    {
                        if (!neuronFeatures.ContainsKey((l, n)))
                            neuronFeatures[(l, n)] = new List<double>();
                        neuronFeatures[(l, n)].Add(network.neurons[l][n]);
                    }
                }
            }

            // 2. Добавляем влияние на R/G/B (корреляцию)
            var finalFeatures = new Dictionary<(int, int), List<double>>();
            foreach (var key in neuronFeatures.Keys)
            {
                var activations = neuronFeatures[key];
                var rCorr = Correlation(activations, outputs.Select(o => o.R).ToList());
                var gCorr = Correlation(activations, outputs.Select(o => o.G).ToList());
                var bCorr = Correlation(activations, outputs.Select(o => o.B).ToList());

                finalFeatures[key] = new List<double> { rCorr, gCorr, bCorr };
            }

            return finalFeatures;
        }

        static double Correlation(List<double> x, List<double> y)
        {
            if (x.Count != y.Count)
                throw new ArgumentException("Lists must have the same length.");

            int n = x.Count;
            if (n == 0)
                return double.NaN;

            // Вычисляем средние значения
            double meanX = x.Average();
            double meanY = y.Average();

            // Вычисляем числитель (ковариация)
            double covariance = 0.0;
            // Вычисляем знаменатели (стандартные отклонения)
            double stdDevX = 0.0;
            double stdDevY = 0.0;

            for (int i = 0; i < n; i++)
            {
                double diffX = x[i] - meanX;
                double diffY = y[i] - meanY;
                covariance += diffX * diffY;
                stdDevX += diffX * diffX;
                stdDevY += diffY * diffY;
            }

            // Проверка деления на ноль
            if (stdDevX == 0 || stdDevY == 0)
                return double.NaN;

            // Корреляция Пирсона
            return covariance / (Math.Sqrt(stdDevX) * Math.Sqrt(stdDevY));
        }

        static void TestLearImage(Network network, Teacher teacher, int iterations)
        {
            Bitmap img = new Bitmap("D:\\_Polygon\\image.png");
            double[][] input = new double[img.Width * img.Height][];
            double[][] output = new double[img.Width * img.Height][];
            int j = 0;
            for (int x = 0; x < img.Width; x++)
            {
                for (int y = 0; y < img.Height; y++)
                {
                    System.Drawing.Color c = img.GetPixel(x, y);
                    input[j] = new double[] { (double)x / img.Width, (double)y / img.Height };
                    output[j] = new double[] { c.R / 255D, c.G / 255D, c.B / 255D };
                    j++;
                }
            }

            for (int i = 516; i < iterations; i++)
            {
                var error = teacher.RunEpoche(input, output);
                Console.WriteLine(error);


                if (i % 5 == 0 || i == iterations - 1)
                {
                    Console.WriteLine(i);
                    Bitmap img2 = new Bitmap(img.Width, img.Height);
                    for (int x = 0; x < img.Width; x++)
                    {
                        for (int y = 0; y < img.Height; y++)
                        {
                            double[] C = network.Compute(new double[] { (double)x / img.Width, (double)y / img.Height });
                            System.Drawing.Color c = System.Drawing.Color.FromArgb((byte)(C[0] * 255D), (byte)(C[1] * 255D), (byte)(C[2] * 255D));
                            img2.SetPixel(x, y, c);
                        }
                    }

                    img2.Save($"sequnce/{i}.png");
                    network.SaveWeights("save.bin");
                }

            }
        }
    }
}