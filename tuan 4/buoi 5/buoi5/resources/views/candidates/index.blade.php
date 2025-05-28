<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

</head>
<body>
    <div class="container">
    <h2>Danh sách ứng viên</h2>
    <table class="table table-bordered">
        <thead>
            <tr>
                <th>Họ tên</th>
                <th>Email</th>
                <th>Ngày sinh</th>
                <th>Mô tả</th>
            </tr>
        </thead>
        <tbody>
            @foreach($datas as $key)
            <tr>
                <td>{{ $key->name }}</td>
                <td>{{ $key->email }}</td>
                <td>{{ $key->birthday }}</td>
                <td>{{ $key->bio }}</td>
            </tr>
            @endforeach
        </tbody>
    </table>
   
</div>

    
</body>
</html>