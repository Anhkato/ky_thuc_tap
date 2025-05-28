<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\Http;
use App\Models\Product;

class ImportProducts extends Command
{
    protected $signature = 'import:products';
    protected $description = 'Import products from fakestoreapi';

    public function handle()
    {
        $response = Http::get('https://fakestoreapi.com/products');
        if ($response->ok()) {
            $products = $response->json();
            foreach ($products as $item) {
                Product::updateOrCreate(
                    ['name' => $item['title']],
                    [
                        'description' => $item['description'],
                        'price' => $item['price'],
                        'image_url' => $item['image'],
                    ]
                );
            }
            $this->info('Products imported successfully!');
        } else {
            $this->error('Failed to fetch products');
        }
    }
}
